//! `marginal`'s tests, kept in a file of their own: with the tests task 158
//! added for its mutation survivors, `marginal.rs` passed the repository's
//! 250 KB cap for a source file (`tests/test_repo_hygiene.py`).

use super::*;

/// A state whose vectors are not the cfg's is refused, where it loaded
/// and panicked on the first `step` (review 2026-09-18, B3).
#[test]
fn a_state_of_the_wrong_shape_is_refused() {
    use crate::{ModelState, OnlineModel, StateError};
    let m = Marginal::new(cfg(2, 1)).unwrap();
    let mut s = m.state();
    let ModelState::Marginal(inner) = &mut s.model else {
        unreachable!()
    };
    inner.mx.pop();
    match Marginal::restore(&s) {
        Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
        other => panic!("{other:?}"),
    }
}

/// A state's configuration is held to what `new` holds a fresh one to:
/// `lags = [0]`, edited into the cfg and the lag ring alike, passed the
/// shape check -- which bounds the ring by `lags.last()`, so an empty ring
/// fit -- and the first learned row took `back = depth - lag = 0` and
/// indexed an empty ring (review 2026-10-06, CD14). A lag of 0 is the pair
/// itself, which `new` refuses.
#[test]
fn a_restored_marginal_whose_cfg_new_refuses_is_refused() {
    use crate::{OnlineModel, State, StateError};
    let mut c = cfg(1, 1);
    c.lags = vec![1];
    let mut m = Marginal::new(c).unwrap();
    for i in 0..5 {
        let v = i as f64;
        OnlineModel::step(&mut m, &[v], &[Some(v)], step_clock(i), 1.0);
    }
    OnlineModel::clear_lags(&mut m);
    let mut v = serde_json::to_value(&m).unwrap();
    v["cfg"]["lags"] = serde_json::json!([0]);
    v["lag"]["lags"] = serde_json::json!([0]);
    let edited: Marginal = serde_json::from_value(v).unwrap();
    assert!(
        Marginal::new(edited.cfg.clone()).is_err(),
        "`new` refuses lags = [0]"
    );
    let s = State::new(crate::ModelState::Marginal(Box::new(edited)));
    match Marginal::restore(&s) {
        Err(StateError::Invalid(e)) => {
            assert!(
                e.contains("configuration") && e.contains("lags must be >= 1"),
                "{e}"
            );
        }
        Ok(mut back) => {
            OnlineModel::step(&mut back, &[1.0], &[Some(1.0)], 1.0, 1.0);
            panic!("lags = [0] loaded and was read");
        }
        Err(e) => panic!("{e}"),
    }
}
use crate::{EwCov, EwCovCfg, EwCovModel, EwCovStat};

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn cfg(p: usize, t: usize) -> MarginalCfg {
    MarginalCfg {
        n_features: p,
        n_targets: t,
        decay: Decay::Halflife(20.0),
        min_weight: vec![0.0; t],
        lags: Vec::new(),
        serial_rule: None,
        cross_lags: None,
        bins: None,
        feature_moments: FeatureMomentLayout::PerTarget,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

/// A windowed marginal over one stream: feature 0 moves; feature 1 and
/// the target are what `x1` and `y` give row `i`; `weight` and `present`
/// say how each row is learned.
fn windowed_pairs(
    window: f64,
    h: f64,
    n: usize,
    x1: impl Fn(usize, f64) -> f64,
    y: impl Fn(usize, f64, f64) -> Option<f64>,
    weight: impl Fn(usize) -> f64,
) -> Marginal {
    let mut c = cfg(2, 1);
    c.decay = Decay::Halflife(h);
    c.window = Some(window);
    let mut m = Marginal::new(c).unwrap();
    let mut s = 11u64;
    for i in 0..n {
        let (a, b) = (lcg(&mut s), lcg(&mut s));
        let d = if i == 0 { 0.0 } else { 1.0 };
        m.step(&[b, x1(i, a)], &[y(i, a, b)], d, weight(i));
    }
    m
}

/// PLAN task 94, for the pairs. A feature, or a target, that holds one
/// value over every row inside a window has no spread there, and the
/// windowed pair says so exactly: zero variance and covariance, the value
/// as its mean, no correlation, and a slope of zero on a target that
/// holds. The subtraction left a remainder that grows with the level and
/// the rows since the boundary, and `beta` divided it by itself. Found by
/// the sweep of every running mean for task 101. The slot that moves
/// keeps its spread.
#[test]
fn a_slot_held_over_the_window_has_no_spread_there() {
    for level in [0.0, 1e3, 1e6, -1e8] {
        for (window, h) in [(0.5, 10.0), (9.0, 10.0), (30.0, 8.0), (199.0, 70.0)] {
            for before in [0usize, 3] {
                for held_feature in [true, false] {
                    let n = 400 + window as usize;
                    let from = n - 1 - window as usize - before;
                    let m = windowed_pairs(
                        window,
                        h,
                        n,
                        |i, a| {
                            if held_feature && i >= from {
                                level + 0.37
                            } else {
                                level + 1e-3 * a
                            }
                        },
                        |i, a, b| {
                            Some(if !held_feature && i >= from {
                                level + 0.25
                            } else {
                                level + 1.0 + 0.5 * a + b
                            })
                        },
                        |_| 1.0,
                    );
                    let case = format!(
                        "level {level}, window {window}, h {h}, from {before} before, \
                             held feature {held_feature}"
                    );
                    let pair = m.pair(0, 1);
                    assert_eq!(pair.cov, 0.0, "{case}: covariance");
                    assert!(pair.corr.is_nan(), "{case}: correlation {}", pair.corr);
                    if held_feature {
                        assert_eq!(pair.var_x, 0.0, "{case}: variance");
                        assert_eq!(pair.mean_x, level + 0.37, "{case}: mean");
                        assert!(pair.beta.is_nan(), "{case}: slope {}", pair.beta);
                    } else {
                        assert_eq!(pair.var_y, 0.0, "{case}: variance");
                        assert_eq!(pair.mean_y, level + 0.25, "{case}: mean");
                        if window >= 1.0 {
                            assert_eq!(pair.beta, 0.0, "{case}: slope");
                        }
                    }
                    if window >= 1.0 {
                        let moving = m.pair(0, 0);
                        assert!(moving.var_x > 0.0, "{case}: the moving feature's spread");
                    }
                }
            }
        }
    }
}

/// One rule says what is nothing, the window's: a window weighing no more
/// than `EMPTY_FRACTION` of the live weight is empty. A run has no rule of
/// its own, so every row with a weight is a learned row for it (review
/// 2026-10-05, CE3): a boundary row at a weight of 1e-12, carrying another
/// value before the run begins, ends the run, and the window reads the
/// spread that row brings from the subtraction, against the definition.
/// It read as held, the spread dropped, where the window's weight already
/// counted the row.
#[test]
fn a_boundary_row_of_little_weight_ends_a_run() {
    let (window, n) = (9.0, 300);
    let boundary = n - 1 - window as usize;
    let value = |i: usize, a: f64| match i {
        _ if i == boundary => -5.0,
        _ if i > boundary => 0.37,
        _ => 1e-3 * a,
    };
    let weight = |i: usize| if i == boundary { 1e-12 } else { 1.0 };
    let m = windowed_pairs(
        window,
        10.0,
        n,
        value,
        |_, a, b| Some(1.0 + 0.5 * a + b),
        weight,
    );
    let pair = m.pair(0, 1);
    // The definition: the rows inside the window, at their decayed weights.
    let lam = Decay::Halflife(10.0).factor(1.0);
    let inside: Vec<(f64, f64)> = (boundary..n)
        .map(|i| (value(i, 0.0), weight(i) * lam.powi((n - 1 - i) as i32)))
        .collect();
    let w: f64 = inside.iter().map(|r| r.1).sum();
    let mean = inside.iter().map(|r| r.0 * r.1).sum::<f64>() / w;
    let var = inside
        .iter()
        .map(|r| r.1 * (r.0 - mean).powi(2))
        .sum::<f64>()
        / w;
    assert!(var > 1e-12, "the case needs a spread: {var:e}");
    assert!(pair.var_x > 0.0, "the row ended the run: {:e}", pair.var_x);
    assert!(
        (pair.var_x - var).abs() <= 1e-3 * var,
        "{:e} against {var:e}",
        pair.var_x
    );
    assert!(
        (pair.mean_x - mean).abs() <= 1e-12,
        "{} against {mean}",
        pair.mean_x
    );
}

/// A window whose rows are each lighter than `EMPTY_FRACTION` of the
/// history, and together heavier, is a window with rows in it, and the
/// pair reads their moments (review 2026-10-05, CE3). The runs counted a
/// row only above that fraction of the target's weight, so they saw none
/// of these, and the pair took the last heavy row's value as held over the
/// window: its `x` as the mean, no variance and no covariance. A thousand
/// unit rows, then twelve at `1e-9`, a window of 10 clock units and no
/// decay: the eleven rows inside, against the definition, to the digits a
/// window of `1.1e-11` of the weight leaves the subtraction, about five.
#[test]
fn a_window_of_light_rows_reads_their_moments() {
    let mut c = cfg(1, 1);
    c.decay = Decay::Halflife(f64::INFINITY);
    c.window = Some(10.0);
    let mut m = Marginal::new(c).unwrap();
    let mut s = 5u64;
    let mut rows = Vec::new();
    for i in 0..1012 {
        let x = lcg(&mut s);
        let y = 0.7 * x + lcg(&mut s);
        let w = if i < 1000 { 1.0 } else { 1e-9 };
        m.step(&[x], &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, w);
        rows.push((x, y));
    }
    let inside = &rows[rows.len() - 11..];
    let n = inside.len() as f64;
    let mx = inside.iter().map(|r| r.0).sum::<f64>() / n;
    let my = inside.iter().map(|r| r.1).sum::<f64>() / n;
    let vx = inside.iter().map(|r| (r.0 - mx).powi(2)).sum::<f64>() / n;
    let vy = inside.iter().map(|r| (r.1 - my).powi(2)).sum::<f64>() / n;
    let cov = inside.iter().map(|r| (r.0 - mx) * (r.1 - my)).sum::<f64>() / n;
    let p = m.pair(0, 0);
    let near = |got: f64, want: f64, scale: f64| (got - want).abs() <= 1e-3 * scale;
    assert!(near(p.n_eff, 11e-9, 11e-9), "weight {:e}", p.n_eff);
    assert!(near(p.mean_x, mx, 1.0), "mean_x {} against {mx}", p.mean_x);
    assert!(near(p.mean_y, my, 1.0), "mean_y {} against {my}", p.mean_y);
    assert!(near(p.var_x, vx, vx), "var_x {} against {vx}", p.var_x);
    assert!(near(p.var_y, vy, vy), "var_y {} against {vy}", p.var_y);
    assert!(
        near(p.cov, cov, (vx * vy).sqrt()),
        "cov {} against {cov}",
        p.cov
    );
}

/// Kish's size inside a window is `W_R² / Q_R`, both remainders of a
/// subtraction, and `Q_R` loses its digits first: a window of light rows
/// holds the squares of their weights. Where it keeps some, the size is the
/// definition's, the rows inside the window, to the digits kept -- three at
/// rows of `1e-5`, whose squares are `1e-10` against a sum of 1000; where
/// the subtraction has none left -- `Q_R` within `64 ε` of the sum it came
/// from -- it is NaN, not the `inf` of a remainder of 0 or a number made of
/// rounding (review 2026-10-05, CE6: 2% off with the window at 1e-8 of the
/// history, `inf` at 1e-11). A thousand unit rows, then twelve light ones,
/// a window of 10 clock units and no decay.
#[test]
fn kish_size_inside_a_window_is_the_rows_inside_or_nothing() {
    for (light, digits) in [(1e-3, true), (1e-5, true), (1e-9, false), (1e-12, false)] {
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(f64::INFINITY);
        c.window = Some(10.0);
        let mut m = Marginal::new(c).unwrap();
        let mut s = 5u64;
        for i in 0..1012 {
            let x = lcg(&mut s);
            let w = if i < 1000 { 1.0 } else { light };
            m.step(
                &[x],
                &[Some(0.7 * x + lcg(&mut s))],
                if i == 0 { 0.0 } else { 1.0 },
                w,
            );
        }
        let got = m.pair(0, 0).n_kish;
        if digits {
            // Eleven rows of one weight: `(11 w)² / (11 w²) = 11`.
            assert!((got - 11.0).abs() <= 1e-3 * 11.0, "{light}: {got}");
        } else {
            assert!(got.is_nan(), "{light}: {got}");
        }
    }
}

/// The floor is `64 ε` of the live sum, and exclusive: a remainder of
/// exactly that is no size, and one of twice that a size. The Kish sums are
/// set by hand, the live one at 2 and every snapshot's below it by the
/// remainder, so both subtractions are exact and the floor is `2^-45`.
#[test]
fn the_windowed_kish_floor_is_sixty_four_epsilons_of_the_live_sum() {
    for (left, kept) in [(2f64.powi(-45), false), (2f64.powi(-44), true)] {
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(f64::INFINITY);
        c.window = Some(10.0);
        let mut m = Marginal::new(c).unwrap();
        let mut s = 5u64;
        for i in 0..30 {
            let x = lcg(&mut s);
            let y = 0.7 * x + lcg(&mut s);
            m.step(&[x], &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        for snap in m.win.as_mut().unwrap().snaps.iter_mut() {
            snap.qt[0] = 2.0 - left;
        }
        m.qt[0] = 2.0;
        let got = m.pair(0, 0).n_kish;
        assert_eq!(got.is_nan(), !kept, "{left:e}: {got}");
    }
}

/// A row without the target ages its runs as it ages its weight, so a run
/// that began inside the window is not credited with weight it no longer
/// carries: a feature that moved inside the window keeps its spread there
/// when the target is present on every other row.
#[test]
fn a_row_without_the_target_ages_its_runs() {
    let (window, n) = (30.0, 400);
    let m = windowed_pairs(
        window,
        3.0,
        n,
        |i, a| if i >= n - 20 { 1e3 + 0.37 } else { 1e3 + a },
        |i, a, b| (i % 2 == 0).then_some(1.0 + 0.5 * a + b),
        |_| 1.0,
    );
    let pair = m.pair(0, 1);
    assert!(
        pair.var_x > 0.0,
        "the feature moved inside the window: {}",
        pair.var_x
    );
    assert!(pair.cov != 0.0);
}

/// A window that holds all but 2^-60 of the weight reads what no window
/// reads, field for field, to the rounding of the subtraction: the
/// weight, Kish's size, the means and the three second moments.
#[test]
fn a_window_holding_the_weight_reads_what_no_window_does() {
    let n = 400;
    let windowed = windowed_pairs(60.0, 1.0, n, |_, a| 3.0 + a, |_, a, b| Some(a - b), |_| 1.0);
    let mut c = cfg(2, 1);
    c.decay = Decay::Halflife(1.0);
    let mut plain = Marginal::new(c).unwrap();
    let mut s = 11u64;
    for i in 0..n {
        let (a, b) = (lcg(&mut s), lcg(&mut s));
        plain.step(
            &[b, 3.0 + a],
            &[Some(a - b)],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    for j in 0..2 {
        let (w, p) = (windowed.pair(0, j), plain.pair(0, j));
        for (what, got, want) in [
            ("weight_sum", w.n_eff, p.n_eff),
            ("n_kish", w.n_kish, p.n_kish),
            ("mean_x", w.mean_x, p.mean_x),
            ("mean_y", w.mean_y, p.mean_y),
            ("var_x", w.var_x, p.var_x),
            ("var_y", w.var_y, p.var_y),
            ("cov", w.cov, p.cov),
        ] {
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1e-300),
                "feature {j}: {what} {got} windowed, {want} without"
            );
        }
    }
}

/// A row of weight 0 takes no step in a pair's means, to the bit: the
/// rows 0.7 and 5.292162135665459 at unit weight and no decay leave each
/// mean with a low part of a whole rounding step, where a zero step would
/// round the double up (`crate::comp::add`).
#[test]
fn a_row_of_no_weight_leaves_the_means_as_they_were() {
    let mut c = cfg(1, 1);
    c.decay = Decay::Halflife(f64::INFINITY);
    let mut m = Marginal::new(c).unwrap();
    for v in [0.7, 5.292162135665459] {
        m.step(&[v], &[Some(v)], 1.0, 1.0);
    }
    assert_eq!(
        (m.mx[0], m.my[0]),
        (2.996081067832729, 2.996081067832729),
        "the fixture"
    );
    assert_eq!(
        m.mx_lo[0], 4.440892098500626e-16,
        "a whole step below the double"
    );
    m.step(&[1.0], &[Some(1.0)], 1.0, 0.0);
    assert_eq!((m.mx[0], m.my[0]), (2.996081067832729, 2.996081067832729));
    assert_eq!(
        (m.mx_lo[0], m.my_lo[0]),
        (4.440892098500626e-16, 4.440892098500626e-16)
    );
}

/// A row without a target ages that target's runs and no other's, over
/// its own slots and no other's: two features held over a window, and
/// two targets each present on half the rows, so that every pair reads
/// its feature as held. A run aged on the wrong rows, or the wrong
/// slots, falls short of the window's weight.
#[test]
fn a_row_without_a_target_ages_that_targets_runs_alone() {
    let (window, n) = (30.0, 400);
    let mut c = cfg(2, 2);
    c.decay = Decay::Halflife(3.0);
    c.window = Some(window);
    let mut m = Marginal::new(c).unwrap();
    let mut s = 13u64;
    let from = n - 1 - window as usize - 3;
    for i in 0..n {
        let (a, b) = (lcg(&mut s), lcg(&mut s));
        let x = if i >= from {
            [1e3 + 0.37, 1e3 + 0.25]
        } else {
            [1e3 + a, 1e3 + b]
        };
        let y = [(i % 2 == 1).then_some(a - b), (i % 2 == 0).then_some(a + b)];
        m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    for t in 0..2 {
        for (j, held) in [1e3 + 0.37, 1e3 + 0.25].into_iter().enumerate() {
            let pair = m.pair(t, j);
            assert_eq!(pair.var_x, 0.0, "target {t}, feature {j}: variance");
            assert_eq!(pair.mean_x, held, "target {t}, feature {j}: mean");
        }
    }
}

/// A state written before the means' low parts, and before the runs,
/// loads and steps: each starts at the size of what it belongs to.
#[test]
fn a_state_without_the_low_parts_loads_and_steps() {
    let mut c = cfg(2, 1);
    c.window = Some(9.0);
    let mut m = Marginal::new(c).unwrap();
    let mut s = 3u64;
    for i in 0..40 {
        m.step(
            &[lcg(&mut s), lcg(&mut s)],
            &[Some(lcg(&mut s))],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    let mut old = serde_json::to_value(&m).unwrap();
    for key in ["mx_lo", "my_lo", "x_runs", "y_runs"] {
        assert!(old.as_object_mut().unwrap().remove(key).is_some(), "{key}");
    }
    let mut back: Marginal = serde_json::from_value(old).unwrap();
    assert!(back.mx_lo.is_empty() && back.my_lo.is_empty());
    for i in 0..40 {
        back.step(&[lcg(&mut s), lcg(&mut s)], &[Some(lcg(&mut s))], 1.0, 1.0);
        let p = back.pair(0, i % 2);
        assert!(p.mean_x.is_finite() && p.var_x.is_finite(), "row {i}");
    }
    assert_eq!((back.mx_lo.len(), back.my_lo.len()), (2, 1));
}

/// Under a window, Kish's size is `ew_cov`'s over the same rows: the
/// same weights, the same `Sum w²`, and the same boundary, so the same
/// subtraction.
#[test]
fn the_windowed_kish_size_is_ew_covs() {
    let (window, h) = (30.0, 10.0);
    let mut c = cfg(1, 1);
    c.decay = Decay::Halflife(h);
    c.window = Some(window);
    let mut m = Marginal::new(c).unwrap();
    let mut ec = EwCovModel::new(EwCovCfg {
        n_features: 2,
        decay: Decay::Halflife(h),
        stats: vec![EwCovStat::Mean],
        min_weight: 0.0,
        precision_prior: None,
        mahal_quantiles: Vec::new(),
        pca: 0,
        pca_every: 0.0,
        max_rows_between_pca: u32::MAX,
        lags: Vec::new(),
        window: Some(window),
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap();
    let mut s = 5u64;
    for i in 0..200 {
        let (x, y) = (lcg(&mut s), lcg(&mut s));
        let d = if i == 0 { 0.0 } else { 1.0 };
        crate::OnlineModel::step(&mut m, &[x], &[Some(y)], d, 1.0);
        crate::OnlineModel::step(&mut ec, &[x, y], &[], d, 1.0);
        if i >= 100 {
            let want = ec.windowed_cov().n_kish().unwrap();
            let got = m.pair(0, 0).n_kish;
            assert!(
                (got - want).abs() <= 1e-12 * want,
                "row {i}: {got} against ew_cov's {want}"
            );
        }
    }
}

/// A stream with three features and three targets that reaches every
/// lag event: serially dependent series, targets absent on a fifth of
/// the rows, unequal weights with zeros, and the ring cleared once, as a
/// session change clears it. The lags `[1, 2, 5, 10]`, with the cross
/// terms kept where `cross_lags` says.
fn cross_lag_run(cross_lags: Option<Vec<usize>>) -> Marginal {
    let mut c = cfg(3, 3);
    c.decay = Decay::Halflife(40.0);
    c.lags = vec![1, 2, 5, 10];
    c.serial_rule = Some(SerialRule::Geometric);
    c.cross_lags = cross_lags;
    let mut m = Marginal::new(c).unwrap();
    let mut seed = 77u64;
    let (mut x, mut y) = ([0.0_f64; 3], [0.0_f64; 3]);
    for i in 0..600 {
        for v in x.iter_mut() {
            *v = 0.8 * *v + lcg(&mut seed);
        }
        let ys: Vec<Option<f64>> = (0..3)
            .map(|t| {
                y[t] = 0.7 * y[t] + 0.5 * x[t] + lcg(&mut seed);
                (lcg(&mut seed) > -0.6).then_some(y[t])
            })
            .collect();
        let w = if i % 11 == 4 {
            0.0
        } else {
            0.5 + (lcg(&mut seed) + 1.0) / 2.0
        };
        if i == 300 {
            OnlineModel::clear_lags(&mut m);
        }
        OnlineModel::step(&mut m, &x, &ys, step_clock(i), w);
    }
    m
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Each lagged correlation is its lagged covariance over the two
/// contemporaneous standard deviations, the autocorrelations over one
/// of them squared -- `ew_cov`'s expression -- to the bit, at every lag
/// and every cross lag.
#[test]
fn a_lagged_correlation_is_its_covariance_over_the_two_deviations() {
    for cross in [None, Some(vec![2, 10])] {
        let m = cross_lag_run(cross.clone());
        let lag = m.lag.as_ref().unwrap();
        for t in 0..3 {
            for j in 0..3 {
                let q = m.pair(t, j);
                let (sx, sy) = (q.var_x.sqrt(), q.var_y.sqrt());
                assert!(sx > 0.0 && sy > 0.0);
                for li in 0..4 {
                    assert_eq!(
                        q.lagcorr_xx[li].to_bits(),
                        (lag.cxx(li, t, j) / (sx * sx)).to_bits()
                    );
                    assert_eq!(
                        q.lagcorr_yy[li].to_bits(),
                        (lag.cyy(li, t) / (sy * sy)).to_bits()
                    );
                }
                for ci in 0..lag.cross_lags().len() {
                    assert_eq!(
                        q.lagcorr_xy[ci].to_bits(),
                        (lag.cxy(ci, t, j) / (sx * sy)).to_bits()
                    );
                    assert_eq!(
                        q.lagcorr_yx[ci].to_bits(),
                        (lag.cyx(ci, t, j) / (sx * sy)).to_bits()
                    );
                }
            }
        }
    }
}

/// A feature constant on every row its target was present on, but not on
/// a row between them where the target was absent: the ring is shared,
/// so the target now against the feature a row back sees that row, and
/// the lagged covariance is not zero where the feature's spread is. Its
/// correlation is undefined, NaN, not the infinity the division gives.
#[test]
fn a_lagged_correlation_over_no_spread_is_nan() {
    let mut c = cfg(1, 1);
    c.lags = vec![1];
    let mut m = Marginal::new(c).unwrap();
    let mut seed = 3u64;
    for i in 0..40 {
        // Odd rows carry the target with the feature at 2; even rows
        // leave the target out and move the feature.
        let (x, y) = if i % 2 == 1 {
            (2.0, Some(lcg(&mut seed)))
        } else {
            (lcg(&mut seed), None)
        };
        OnlineModel::step(&mut m, &[x], &[y], step_clock(i), 1.0);
    }
    let q = m.pair(0, 0);
    let lag = m.lag.as_ref().unwrap();
    assert_eq!(q.var_x, 0.0, "the feature is constant on the target's rows");
    assert!(q.var_y > 0.0);
    assert!(lag.cyx(0, 0, 0) != 0.0, "the ring saw the rows between");
    assert!(q.lagcorr_yx[0].is_nan(), "{}", q.lagcorr_yx[0]);
    assert!(q.lagcorr_xx[0].is_nan() && q.lagcorr_xy[0].is_nan());
}

/// E70 (docs/PLAN.md task 123): `cross_lags` chooses which cross terms
/// are kept and touches nothing else. The autocorrelations, `n_serial`,
/// `t_serial` and the fitted decays are the same to the bit whether the
/// cross terms are kept at every lag, at some, or at none.
#[test]
fn cross_lags_leave_the_serial_correction_to_the_bit() {
    let all = cross_lag_run(None);
    let mut corrected = 0;
    for cross in [vec![1], vec![], vec![2, 10]] {
        let some = cross_lag_run(Some(cross.clone()));
        for t in 0..3 {
            for j in 0..3 {
                let (a, b) = (all.pair(t, j), some.pair(t, j));
                let why = format!("cross_lags {cross:?}, target {t}, feature {j}");
                assert_eq!(bits(&a.lagcorr_xx), bits(&b.lagcorr_xx), "{why}");
                assert_eq!(bits(&a.lagcorr_yy), bits(&b.lagcorr_yy), "{why}");
                let serial = |q: &Pair| [q.n_serial, q.t_serial, q.phi_x, q.phi_y, q.corr, q.t];
                assert_eq!(bits(&serial(&a)), bits(&serial(&b)), "{why}");
                assert_eq!(a.lagcorr_xy.len(), 4, "{why}: every lag by default");
                assert_eq!(b.lagcorr_xy.len(), cross.len(), "{why}");
                assert_eq!(b.lagcorr_yx.len(), cross.len(), "{why}");
                corrected += usize::from(b.n_serial.is_finite());
            }
        }
    }
    // The correction was computed, not NaN on both sides alike.
    assert_eq!(corrected, 27, "every pair's n_serial is a number");
}

/// ... and the cross terms it keeps are the default's at those lags, to
/// the bit: the accumulators, not only the correlations read from them.
/// Every lag named outright takes the general path and the default its
/// own loop (`MarginalLags::update_target`), so this is also what holds
/// the two loops to the same bits.
#[test]
fn cross_lags_keep_the_default_cross_terms_at_their_lags() {
    let all = cross_lag_run(None);
    let la = all.lag.as_ref().unwrap();
    for cross in [vec![1, 5], vec![1, 2, 5, 10]] {
        let some = cross_lag_run(Some(cross.clone()));
        let ls = some.lag.as_ref().unwrap();
        assert_eq!(ls.cross_lags(), cross.as_slice());
        let at: Vec<usize> = cross
            .iter()
            .map(|l| [1, 2, 5, 10].iter().position(|m| m == l).unwrap())
            .collect();
        for t in 0..3 {
            for j in 0..3 {
                for (ci, &li) in at.iter().enumerate() {
                    assert_eq!(ls.cxy(ci, t, j).to_bits(), la.cxy(li, t, j).to_bits());
                    assert_eq!(ls.cyx(ci, t, j).to_bits(), la.cyx(li, t, j).to_bits());
                }
                for li in 0..4 {
                    assert_eq!(ls.cxx(li, t, j).to_bits(), la.cxx(li, t, j).to_bits());
                    assert_eq!(ls.cyy(li, t).to_bits(), la.cyy(li, t).to_bits());
                }
                let (a, b) = (all.pair(t, j), some.pair(t, j));
                let pick = |v: &[f64]| at.iter().map(|&li| v[li]).collect::<Vec<_>>();
                assert_eq!(bits(&b.lagcorr_xy), bits(&pick(&a.lagcorr_xy)));
                assert_eq!(bits(&b.lagcorr_yx), bits(&pick(&a.lagcorr_yx)));
            }
        }
    }
}

#[test]
fn a_bad_cross_lags_is_refused_by_name() {
    for (lags, cross, msg) in [
        (
            vec![1, 2, 5],
            vec![3],
            "cross_lags must each be one of lags, and 3 is not",
        ),
        (
            vec![1, 2, 5],
            vec![1, 0],
            "cross_lags must be strictly increasing",
        ),
        (
            vec![1, 2, 5],
            vec![5, 1],
            "cross_lags must be strictly increasing",
        ),
        (
            vec![1, 2, 5],
            vec![2, 2],
            "cross_lags must be strictly increasing",
        ),
        (
            vec![1],
            vec![0],
            "cross_lags must each be one of lags, and 0 is not",
        ),
        (vec![], vec![1], "cross_lags needs `lags`"),
        (vec![], vec![], "cross_lags needs `lags`"),
    ] {
        let mut c = cfg(1, 1);
        c.lags = lags.clone();
        c.cross_lags = Some(cross.clone());
        let err = Marginal::new(c).unwrap_err();
        assert!(
            err.contains(msg),
            "lags {lags:?}, cross_lags {cross:?}: {err}"
        );
    }
}

/// A state written before E70 has no `cross_lags`, in the config or in
/// the lag moments. It loads as cross terms at every lag, which is what
/// it holds, and learns on exactly as the state that wrote it.
#[test]
fn a_state_without_cross_lags_loads_with_every_lag() {
    let mut m = cross_lag_run(None);
    let mut old = serde_json::to_value(&m).unwrap();
    assert!(
        old["cfg"]
            .as_object_mut()
            .unwrap()
            .remove("cross_lags")
            .is_some()
    );
    assert!(
        old["lag"]
            .as_object_mut()
            .unwrap()
            .remove("cross_lags")
            .is_some()
    );
    let mut back: Marginal = serde_json::from_value(old).unwrap();
    assert_eq!(back, m);
    let mut seed = 5u64;
    for i in 0..50 {
        let x = [lcg(&mut seed), lcg(&mut seed), lcg(&mut seed)];
        let y = [Some(lcg(&mut seed)), None, Some(lcg(&mut seed))];
        OnlineModel::step(&mut m, &x, &y, step_clock(i + 1), 1.0);
        OnlineModel::step(&mut back, &x, &y, step_clock(i + 1), 1.0);
    }
    assert_eq!(back, m);
    assert_eq!(back.pair(2, 1).lagcorr_xy.len(), 4);
}

/// E66 test 1: the lagged moments are `ew_cov(lags=)`'s, to the bit. Both
/// centre each leg at the pre-row mean and mix with the same `a`/`b`, so
/// there is no room for them to differ -- and if the two ever drift, one
/// of the two recursions has been changed without the other.
#[test]
fn lagged_pair_moments_are_ew_covs_to_the_bit() {
    let lags = vec![1usize, 3, 7];
    let mut c = cfg(1, 1);
    c.decay = Decay::Halflife(30.0);
    c.lags = lags.clone();
    let mut m = Marginal::new(c).unwrap();

    // `ew_cov` over the same two columns, in the order [x, y]: its lagged
    // co-moment for (0, 1) is `E[dx_t·dy_{t-l}]`, which is `cxy`.
    let mut ec = EwCovModel::new(EwCovCfg {
        n_features: 2,
        decay: Decay::Halflife(30.0),
        stats: vec![EwCovStat::Mean],
        min_weight: 0.0,
        precision_prior: None,
        mahal_quantiles: Vec::new(),
        pca: 0,
        pca_every: 0.0,
        max_rows_between_pca: u32::MAX,
        lags: lags.clone(),
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap();

    let mut seed = 21u64;
    for i in 0..80 {
        let x = lcg(&mut seed) * 2.0;
        let y = 0.6 * x + lcg(&mut seed);
        let d = if i == 0 { 0.0 } else { 1.0 };
        crate::OnlineModel::step(&mut m, &[x], &[Some(y)], d, 1.0);
        crate::OnlineModel::step(&mut ec, &[x, y], &[], d, 1.0);
    }
    let pair = m.pair(0, 0);
    let want = ec.lag().expect("ew_cov has lags");
    for (li, _) in lags.iter().enumerate() {
        // Covariances, not the correlations the pair reports: compare the
        // accumulators themselves so the normalization cannot hide a
        // difference.
        let lag = m.lag.as_ref().unwrap();
        assert_eq!(
            lag.cxx(li, 0, 0).to_bits(),
            want.get(li, 0, 0).to_bits(),
            "lag {li}: the feature's own autocovariance"
        );
        assert_eq!(
            lag.cyy(li, 0).to_bits(),
            want.get(li, 1, 1).to_bits(),
            "lag {li}: the target's"
        );
        assert_eq!(
            lag.cxy(li, 0, 0).to_bits(),
            want.get(li, 0, 1).to_bits(),
            "lag {li}: x now against y back"
        );
        assert_eq!(
            lag.cyx(li, 0, 0).to_bits(),
            want.get(li, 1, 0).to_bits(),
            "lag {li}: y now against x back"
        );
    }
    assert_eq!(pair.lagcorr_xx.len(), lags.len());
}

/// The identity above for every feature of every target: two of each,
/// so a slot read as `t·p + j` is one of four, and `ew_cov` over the
/// four columns `[x0, x1, y0, y1]` has each lagged co-moment by index.
#[test]
fn lagged_pair_moments_are_ew_covs_for_every_feature_and_target() {
    let lags = vec![1usize, 3];
    let (p, nt) = (2usize, 2usize);
    let mut c = cfg(p, nt);
    c.decay = Decay::Halflife(30.0);
    c.lags = lags.clone();
    let mut m = Marginal::new(c).unwrap();
    let mut ec = EwCovModel::new(EwCovCfg {
        n_features: p + nt,
        decay: Decay::Halflife(30.0),
        stats: vec![EwCovStat::Mean],
        min_weight: 0.0,
        precision_prior: None,
        mahal_quantiles: Vec::new(),
        pca: 0,
        pca_every: 0.0,
        max_rows_between_pca: u32::MAX,
        lags: lags.clone(),
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap();
    let mut seed = 23u64;
    for i in 0..80 {
        let x0 = lcg(&mut seed) * 2.0;
        let x1 = lcg(&mut seed) + 0.3 * x0;
        let y0 = 0.6 * x0 + lcg(&mut seed);
        let y1 = -0.4 * x1 + lcg(&mut seed);
        let d = if i == 0 { 0.0 } else { 1.0 };
        crate::OnlineModel::step(&mut m, &[x0, x1], &[Some(y0), Some(y1)], d, 1.0);
        crate::OnlineModel::step(&mut ec, &[x0, x1, y0, y1], &[], d, 1.0);
    }
    let lag = m.lag.as_ref().unwrap();
    let want = ec.lag().expect("ew_cov has lags");
    for li in 0..lags.len() {
        for t in 0..nt {
            assert_eq!(
                lag.cyy(li, t).to_bits(),
                want.get(li, p + t, p + t).to_bits(),
                "lag {li}, target {t}: the target's own"
            );
            for j in 0..p {
                assert_eq!(
                    lag.cxx(li, t, j).to_bits(),
                    want.get(li, j, j).to_bits(),
                    "lag {li}, target {t}, feature {j}: the feature's own"
                );
                assert_eq!(
                    lag.cxy(li, t, j).to_bits(),
                    want.get(li, j, p + t).to_bits(),
                    "lag {li}, target {t}, feature {j}: x now against y back"
                );
                assert_eq!(
                    lag.cyx(li, t, j).to_bits(),
                    want.get(li, p + t, j).to_bits(),
                    "lag {li}, target {t}, feature {j}: y now against x back"
                );
            }
        }
    }
}

/// The runs are a window's, and a model without one keeps none: every
/// slot's run stays unstarted, where under a window each slot's run
/// starts at its first learned row (docs/PERFORMANCE.md §24).
#[test]
fn only_a_window_keeps_the_runs() {
    for window in [None, Some(50.0)] {
        let mut c = cfg(2, 2);
        c.window = window;
        let mut m = Marginal::new(c).unwrap();
        let mut s = 9u64;
        for i in 0..10 {
            let x = [lcg(&mut s), 0.25];
            m.step(&x, &[Some(lcg(&mut s)), Some(1.0)], step_clock(i), 1.0);
        }
        assert_eq!(m.rows_t, vec![10, 10], "the counts are kept either way");
        // Not even allocated without one (docs/PLAN.md task 128).
        assert_eq!(m.x_runs.is_off(), window.is_none());
        assert_eq!(m.y_runs.is_off(), window.is_none());
        let started = |r: &crate::Runs, k: usize| {
            (0..k)
                .filter(|&i| r.started_by(k, i, u64::MAX).is_some())
                .count()
        };
        let (xs, ys) = (started(&m.x_runs, 4), started(&m.y_runs, 2));
        if window.is_some() {
            assert_eq!((xs, ys), (4, 2), "every slot has a run under a window");
            assert_eq!(m.x_runs.started_by(4, 1, 1), Some(0.25), "the held feature");
        } else {
            assert_eq!((xs, ys), (0, 0), "no run without one");
        }
    }
}

/// Each target counts its own learned rows: a row of weight 0 counts for
/// none, and any other row for every target it carries, however light
/// next to the target's weight -- the window alone says what is nothing
/// (`EwCov::update`; review 2026-10-05, CE3, where a row at
/// `EMPTY_FRACTION` of the target's weight counted for none).
#[test]
fn each_target_counts_its_learned_rows() {
    let mut m = Marginal::new(cfg(1, 2)).unwrap();
    for i in 0..20 {
        let y0 = (i % 2 == 0).then_some(1.0);
        let w = if i % 5 == 4 { 0.0 } else { 1.0 };
        m.step(&[0.5], &[y0, Some(2.0)], if i == 0 { 0.0 } else { 1.0 }, w);
    }
    assert_eq!(m.rows_t, vec![8, 16]);
    let lam = m.cfg.decay.factor(1.0);
    let crumb = crate::window::EMPTY_FRACTION * (lam * m.wt[0]);
    m.step(&[0.5], &[Some(1.0), Some(2.0)], 1.0, crumb);
    assert_eq!(m.rows_t, vec![9, 17], "a row at the fraction counts");
    m.step(&[0.5], &[Some(1.0), Some(2.0)], 1.0, 1e-300);
    assert_eq!(m.rows_t, vec![10, 18], "and one far below it");
    m.step(&[0.5], &[None, Some(2.0)], 1.0, 0.0);
    assert_eq!(m.rows_t, vec![10, 18], "a row of weight 0 does not");
}

/// A standard normal from the module's LCG, by Box-Muller.
fn normal(state: &mut u64) -> f64 {
    let u1 = (lcg(state) + 1.0) / 2.0;
    let u2 = (lcg(state) + 1.0) / 2.0;
    let u1 = u1.max(1e-12);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// A truncated factor at or below zero -- two series whose
/// autocorrelations have opposite signs, `1 + 2·0.8·(−0.8) = −0.28` -- is
/// an estimate outside the parameter space, and says so with NaN, the
/// answer `Geometric` gives a factor it cannot form. It was floored at
/// `f64::MIN_POSITIVE`, so `n_serial` came out `+inf` and `t_serial`
/// `±inf`: infinite evidence from a correction that had failed (review
/// 2026-09-12, S18).
#[test]
fn a_truncated_factor_that_is_not_positive_is_nan() {
    for (rx, ry) in [(0.8, -0.8), (0.5, -1.0)] {
        let (f, px, py) = serial_factor(SerialRule::Truncated, &[1], &[rx], &[ry]);
        assert!(f.is_nan(), "({rx}, {ry}): {f}");
        assert!(px.is_nan() && py.is_nan());
    }
    let (f, _, _) = serial_factor(SerialRule::Truncated, &[1], &[0.8], &[0.8]);
    assert!((f - 2.28).abs() < 1e-12, "{f}");
}

/// `"bartlett"` against its definition: lag `l` at `1 − l/(L + 1)`, `L`
/// the longest kept lag, sparse lags weighted by their own `l`. On S18's
/// pair, `+0.8` and `−0.8` at lag 1, the weight halves the product and
/// the factor is `1 − 0.64 = 0.36`, where `"truncated"` has none
/// (docs/PLAN.md task 135); a factor still at or below zero is NaN.
#[test]
fn the_bartlett_factor_is_its_definition() {
    let (lags, rx, ry) = ([1usize, 2, 5], [0.7, 0.4, 0.1], [0.6, -0.2, 0.3]);
    let want = 1.0
        + 2.0 * ((1.0 - 1.0 / 6.0) * 0.42 + (1.0 - 2.0 / 6.0) * -0.08 + (1.0 - 5.0 / 6.0) * 0.03);
    let (f, px, py) = serial_factor(SerialRule::Bartlett, &lags, &rx, &ry);
    assert!((f - want).abs() < 1e-15, "{f} against {want}");
    assert!(px.is_nan() && py.is_nan());
    let (f, _, _) = serial_factor(SerialRule::Bartlett, &[1], &[0.8], &[-0.8]);
    assert!((f - 0.36).abs() < 1e-15, "{f}");
    let (f, _, _) = serial_factor(SerialRule::Truncated, &[1], &[0.8], &[-0.8]);
    assert!(f.is_nan());
    let (f, _, _) = serial_factor(SerialRule::Bartlett, &[1, 2], &[0.9, 0.0], &[-1.0, 0.0]);
    assert!(f.is_nan(), "1 − 2·(2/3)·0.9 < 0: {f}");
    // A lag the estimator could not read is left out, not zeroed twice.
    let (f, _, _) = serial_factor(SerialRule::Bartlett, &[1, 2], &[0.5, f64::NAN], &[0.5, 0.3]);
    assert!((f - (1.0 + 2.0 * (2.0 / 3.0) * 0.25)).abs() < 1e-15, "{f}");
}

/// E66 tests 3 and 4, the ones the feature exists for. Two *independent*
/// AR(1) series, so the true correlation is zero and `t` should be
/// N(0,1) — but it is not: with both series smooth, the sample
/// correlation's variance is `[1 + 2·p/(1−p)]/n` with `p = phi_x·phi_y`,
/// so `t` is over-dispersed by `sqrt` of that bracket. `t_serial` is the
/// same statistic against a count that has been told, and it is the one
/// that comes out standard.
#[test]
fn t_serial_is_standard_where_t_is_over_dispersed() {
    let (phi_x, phi_y) = (0.9_f64, 0.8_f64);
    let reps = 200;
    let n = 1500;
    let prod = phi_x * phi_y;
    let inflation = (1.0 + 2.0 * prod / (1.0 - prod)).sqrt();

    let (mut t_vals, mut ts_vals) = (Vec::new(), Vec::new());
    let (mut px_hat, mut py_hat) = (0.0, 0.0);
    for rep in 0..reps {
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(f64::INFINITY); // lam = 1: the classical result
        c.lags = vec![1, 2, 3, 5, 8, 13];
        c.serial_rule = Some(SerialRule::Geometric);
        let mut m = Marginal::new(c).unwrap();
        let mut seed = 1000 + rep as u64;
        let (mut x, mut y) = (0.0, 0.0);
        for i in 0..n {
            x = phi_x * x + (1.0 - phi_x * phi_x).sqrt() * normal(&mut seed);
            y = phi_y * y + (1.0 - phi_y * phi_y).sqrt() * normal(&mut seed);
            crate::OnlineModel::step(
                &mut m,
                &[x],
                &[Some(y)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let pair = m.pair(0, 0);
        t_vals.push(pair.t);
        ts_vals.push(pair.t_serial);
        px_hat += pair.phi_x / reps as f64;
        py_hat += pair.phi_y / reps as f64;
    }
    let sd = |v: &[f64]| {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        (v.iter().map(|a| (a - mean) * (a - mean)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
    };
    let (sd_t, sd_ts) = (sd(&t_vals), sd(&ts_vals));

    // The fitted decays land on the truth (test 4).
    assert!((px_hat - phi_x).abs() < 0.05, "phi_x {px_hat} vs {phi_x}");
    assert!((py_hat - phi_y).abs() < 0.05, "phi_y {py_hat} vs {phi_y}");
    // `t` is over-dispersed by roughly the Bartlett inflation...
    assert!(
        sd_t > 0.6 * inflation && sd_t < 1.6 * inflation,
        "sd(t) {sd_t} should be near the inflation {inflation}"
    );
    // ... and `t_serial` is standard, which is the whole point.
    assert!(
        (0.75..1.35).contains(&sd_ts),
        "sd(t_serial) {sd_ts} should be near 1 (sd(t) was {sd_t}, inflation {inflation})"
    );
}

/// PLAN §13.4 for `marginal`: one pair, computed directly over the rows
/// inside the window and nothing else. The boundary is inclusive, as
/// `window.rs` states it: a row exactly `window` old is inside.
#[test]
fn a_windowed_pair_is_the_pair_of_the_rows_inside_the_window() {
    let (half_life, window) = (25.0, 70.0);
    let mut c = cfg(1, 1);
    c.decay = Decay::Halflife(half_life);
    c.window = Some(window);
    let mut m = Marginal::new(c).unwrap();

    let mut seed = 99u64;
    let (mut xs, mut ys, mut t, mut clock) = (vec![], vec![], vec![], 0.0);
    let mut on_the_boundary = 0;
    for i in 0..140 {
        // `lcg` is [-1, 1): a clock increment must be its magnitude, or
        // the clock runs backwards and the window means nothing.
        // Quarter units, exact in a double, so some rows land exactly
        // one window old (the boundary is inclusive).
        let d = if i == 0 {
            0.0
        } else {
            0.25 * (2.0 + (lcg(&mut seed).abs() * 8.0).floor())
        };
        clock += d;
        let x = lcg(&mut seed) * 3.0;
        let y = 2.0 * x + lcg(&mut seed) * 0.2;
        crate::OnlineModel::step(&mut m, &[x], &[Some(y)], d, 1.0);
        xs.push(x);
        ys.push(y);
        t.push(clock);

        if i >= 5 {
            let now = clock;
            let keep: Vec<usize> = (0..=i).filter(|&j| now - t[j] <= window).collect();
            on_the_boundary += keep.iter().filter(|&&j| now - t[j] == window).count();
            let w: Vec<f64> = keep
                .iter()
                .map(|&j| 0.5_f64.powf((now - t[j]) / half_life))
                .collect();
            let wsum: f64 = w.iter().sum();
            let mx: f64 = keep.iter().zip(&w).map(|(&j, wi)| wi * xs[j]).sum::<f64>() / wsum;
            let my: f64 = keep.iter().zip(&w).map(|(&j, wi)| wi * ys[j]).sum::<f64>() / wsum;
            let vx: f64 = keep
                .iter()
                .zip(&w)
                .map(|(&j, wi)| wi * (xs[j] - mx) * (xs[j] - mx))
                .sum::<f64>()
                / wsum;
            let cov: f64 = keep
                .iter()
                .zip(&w)
                .map(|(&j, wi)| wi * (xs[j] - mx) * (ys[j] - my))
                .sum::<f64>()
                / wsum;
            let got = m.pair(0, 0);
            let tol = |a: f64, b: f64| (a - b).abs() < 1e-8 * b.abs().max(1.0);
            assert!(
                tol(got.n_eff, wsum),
                "row {i} n_eff: {} vs {wsum}",
                got.n_eff
            );
            assert!(
                tol(got.mean_x, mx),
                "row {i} mean_x: {} vs {mx}",
                got.mean_x
            );
            assert!(
                tol(got.mean_y, my),
                "row {i} mean_y: {} vs {my}",
                got.mean_y
            );
            assert!(tol(got.var_x, vx), "row {i} var_x: {} vs {vx}", got.var_x);
            assert!(tol(got.cov, cov), "row {i} cov: {} vs {cov}", got.cov);
        }
    }
    // The boundary was exercised, not only the rows either side of it.
    assert!(
        on_the_boundary >= 5,
        "{on_the_boundary} reads had a row exactly one window old"
    );
}

/// Review 2026-09-12, C17: the same pair against a feature and a target
/// that sit at `1e8`. `cut` used to go back through the raw moment `s +
/// ma·mb`, which at this level loses the variance and the covariance
/// entirely; the oracle is two-pass, so it is right at any offset. The
/// boundary is inclusive, as `window.rs` states it.
#[test]
fn a_windowed_pair_keeps_its_precision_at_a_large_offset() {
    let (half_life, window) = (25.0, 70.0);
    let mut c = cfg(1, 1);
    c.decay = Decay::Halflife(half_life);
    c.window = Some(window);
    let mut m = Marginal::new(c).unwrap();

    let mut seed = 4242u64;
    let (mut xs, mut ys, mut t, mut clock) = (vec![], vec![], vec![], 0.0);
    let mut on_the_boundary = 0;
    for i in 0..160 {
        // Quarter units, exact in a double, so some rows land exactly
        // one window old (the boundary is inclusive).
        let d = if i == 0 {
            0.0
        } else {
            0.25 * (2.0 + (lcg(&mut seed).abs() * 8.0).floor())
        };
        clock += d;
        let u = lcg(&mut seed) * 3.0;
        let x = 1e8 + u;
        let y = 1e8 + 2.0 * u + lcg(&mut seed) * 0.2;
        crate::OnlineModel::step(&mut m, &[x], &[Some(y)], d, 1.0);
        xs.push(x);
        ys.push(y);
        t.push(clock);

        if i >= 5 {
            let now = clock;
            let keep: Vec<usize> = (0..=i).filter(|&j| now - t[j] <= window).collect();
            on_the_boundary += keep.iter().filter(|&&j| now - t[j] == window).count();
            let w: Vec<f64> = keep
                .iter()
                .map(|&j| 0.5_f64.powf((now - t[j]) / half_life))
                .collect();
            let wsum: f64 = w.iter().sum();
            let mean =
                |v: &[f64]| keep.iter().zip(&w).map(|(&j, wj)| wj * v[j]).sum::<f64>() / wsum;
            let (mx, my) = (mean(&xs), mean(&ys));
            let co = |a: &[f64], ma: f64, b: &[f64], mb: f64| {
                keep.iter()
                    .zip(&w)
                    .map(|(&j, wj)| wj * (a[j] - ma) * (b[j] - mb))
                    .sum::<f64>()
                    / wsum
            };
            let (vx, vy, cov) = (
                co(&xs, mx, &xs, mx),
                co(&ys, my, &ys, my),
                co(&xs, mx, &ys, my),
            );
            let got = m.pair(0, 0);
            let close = |got: f64, want: f64| (got - want).abs() < 1e-6 * want.abs().max(1.0);
            assert!(
                (got.n_eff - wsum).abs() < 1e-9 * wsum,
                "row {i} n_eff: {} vs {wsum}",
                got.n_eff
            );
            assert!(
                (got.mean_x - mx).abs() < 1e-12 * mx,
                "row {i} mean_x: {} vs {mx}",
                got.mean_x
            );
            assert!(
                (got.mean_y - my).abs() < 1e-12 * my,
                "row {i} mean_y: {} vs {my}",
                got.mean_y
            );
            assert!(close(got.var_x, vx), "row {i} var_x: {} vs {vx}", got.var_x);
            assert!(close(got.var_y, vy), "row {i} var_y: {} vs {vy}", got.var_y);
            assert!(close(got.cov, cov), "row {i} cov: {} vs {cov}", got.cov);
        }
    }
    assert!(
        on_the_boundary >= 5,
        "{on_the_boundary} reads had a row exactly one window old"
    );
}

/// A window no stream reaches leaves every pair exactly as it was.
#[test]
fn a_marginal_window_no_stream_reaches_changes_nothing() {
    let mk = |window: Option<f64>| {
        let mut c = cfg(2, 1);
        c.window = window;
        Marginal::new(c).unwrap()
    };
    let (mut plain, mut windowed) = (mk(None), mk(Some(1e9)));
    let mut seed = 4u64;
    for i in 0..60 {
        let x = [lcg(&mut seed), lcg(&mut seed)];
        let y = x[0] - x[1];
        let d = if i == 0 { 0.0 } else { 1.0 };
        crate::OnlineModel::step(&mut plain, &x, &[Some(y)], d, 1.0);
        crate::OnlineModel::step(&mut windowed, &x, &[Some(y)], d, 1.0);
    }
    for j in 0..2 {
        let (a, b) = (plain.pair(0, j), windowed.pair(0, j));
        assert_eq!(a.mean_x.to_bits(), b.mean_x.to_bits(), "feature {j} mean");
        assert_eq!(a.cov.to_bits(), b.cov.to_bits(), "feature {j} cov");
        assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits(), "feature {j} n_eff");
    }
}

/// After a gap that empties the window, the model's `n_eff` is 0 exactly,
/// as every pair's is -- not the rounding crumb the truncating
/// subtraction leaves (review 2026-09-18, S4).
#[test]
fn a_gap_that_empties_the_window_leaves_n_eff_exactly_zero() {
    let (half_life, window) = (25.0, 70.0);
    let mut c = cfg(2, 1);
    c.decay = Decay::Halflife(half_life);
    c.window = Some(window);
    let mut m = Marginal::new(c).unwrap();
    let mut seed = 7u64;
    // Fill the window with real rows.
    for i in 0..40 {
        let x = [lcg(&mut seed), lcg(&mut seed)];
        let y = x[0] - x[1];
        let d = if i == 0 { 0.0 } else { 2.0 };
        crate::OnlineModel::step(&mut m, &x, &[Some(y)], d, 1.0);
    }
    // Advance the clock well past the window with zero-weight rows: the
    // clock moves, nothing is learned, so the window empties.
    for _ in 0..60 {
        crate::OnlineModel::step(&mut m, &[0.0, 0.0], &[Some(0.0)], 2.0, 0.0);
    }
    assert_eq!(m.n_eff(), 0.0, "n_eff is a crumb, not 0");
    assert_eq!(m.target_weight(0), 0.0, "target_weight is a crumb, not 0");
    for j in 0..2 {
        assert_eq!(m.pair(0, j).n_eff, 0.0, "pair {j} n_eff is a crumb, not 0");
    }
}

/// Weighted, decayed moments of one pair written out longhand as sums
/// over the rows the target was present: the oracle the recursion is
/// held to.
struct Longhand {
    xs: Vec<f64>,
    ys: Vec<f64>,
    ws: Vec<f64>,
    lams: Vec<f64>,
}

impl Longhand {
    /// The decay each row has suffered by the end: the product of the
    /// factors of every later row (present or not -- time passes for
    /// them all).
    fn weights(&self) -> Vec<f64> {
        let n = self.ws.len();
        (0..n)
            .map(|i| self.ws[i] * self.lams[i + 1..].iter().product::<f64>())
            .collect()
    }

    fn moments(&self) -> (f64, f64, f64, f64, f64, f64, f64) {
        let w = self.weights();
        let sw: f64 = w.iter().sum();
        let sq: f64 = w.iter().map(|v| v * v).sum();
        let mx = w.iter().zip(&self.xs).map(|(w, x)| w * x).sum::<f64>() / sw;
        let my = w.iter().zip(&self.ys).map(|(w, y)| w * y).sum::<f64>() / sw;
        let sxx = w
            .iter()
            .zip(&self.xs)
            .map(|(w, x)| w * (x - mx) * (x - mx))
            .sum::<f64>()
            / sw;
        let syy = w
            .iter()
            .zip(&self.ys)
            .map(|(w, y)| w * (y - my) * (y - my))
            .sum::<f64>()
            / sw;
        let sxy = w
            .iter()
            .zip(&self.xs)
            .zip(&self.ys)
            .map(|((w, x), y)| w * (x - mx) * (y - my))
            .sum::<f64>()
            / sw;
        (sw, sq, mx, my, sxx, syy, sxy)
    }
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

#[test]
fn validation_rejects_each_bad_field() {
    let err = |c: MarginalCfg, what: &str| {
        let e = Marginal::new(c).unwrap_err();
        assert!(e.contains(what), "{e}");
    };
    err(cfg(0, 1), "at least one feature");
    err(cfg(1, 0), "at least one target");
    let mut c = cfg(1, 1);
    c.min_weight = vec![-1.0];
    err(c.clone(), "min_weight");
    // A gate that never opens, as for every other model (S27).
    c.min_weight = vec![f64::INFINITY];
    assert!(Marginal::new(c.clone()).is_ok());
    c.min_weight = vec![f64::NAN];
    err(c.clone(), "min_weight");
    c.min_weight = vec![1.0, 2.0];
    err(c.clone(), "2 entries for 1 targets");
    c.min_weight = vec![];
    err(c, "0 entries for 1 targets");
    // A window with lags is taken since task 137: the snapshots hold the
    // lag moments, and the spec layer asks for `window_lags` by name.
    let mut c = cfg(1, 1);
    c.window = Some(50.0);
    c.lags = vec![1];
    Marginal::new(c).unwrap();
    Marginal::new(cfg(3, 2)).unwrap();
}

/// A windowed lag moment is the increments made inside the window
/// (docs/PLAN.md task 137). An unwindowed twin's weight and moments,
/// after the row before the window's first and after the last, give it
/// in sum form, `(W·C − f·λ·W_u·C_u) / (W − f·λ·W_u)`, read through
/// `lagcorr` against the window's own variances. The target is missing
/// on every fifth row, where its moments hold and its weight ages.
#[test]
fn a_windowed_lag_moment_is_the_increments_inside_the_window() {
    let (hl, window, n) = (30.0, 50.0, 200usize);
    let mut c = cfg(2, 1);
    c.decay = Decay::Halflife(hl);
    c.lags = vec![1, 2];
    c.cross_lags = Some(vec![1]);
    c.min_weight = vec![0.0];
    let mut live = Marginal::new(c.clone()).unwrap();
    c.window = Some(window);
    let mut win = Marginal::new(c).unwrap();
    let mut s = 11u64;
    let mut hist: Vec<(f64, Vec<f64>, Vec<f64>, f64)> = Vec::new();
    for i in 0..n {
        let x = [lcg(&mut s), lcg(&mut s)];
        let y = (i % 5 != 0).then(|| x[0] + 0.3 * lcg(&mut s));
        let d = if i == 0 { 0.0 } else { 1.0 };
        live.step(&x, &[y], d, 1.0);
        win.step(&x, &[y], d, 1.0);
        let l = live.lag.as_ref().unwrap();
        hist.push((
            live.wt[0],
            (0..2).map(|li| l.cyy(li, 0)).collect(),
            (0..2).map(|li| l.cxx(li, 0, 1)).collect(),
            l.cxy(0, 0, 1),
        ));
    }
    // The last row's clock is n − 1; the window keeps the rows at most
    // `window` behind it, so the snapshot is the one before row `u`.
    let lam = 0.5f64.powf(1.0 / hl);
    let u = n - 1 - window as usize;
    let f = 0.5f64.powf((n - 1 - u) as f64 / hl);
    let (w_now, yy_now, xx_now, xy_now) = &hist[n - 1];
    let (w_then, yy_then, xx_then, xy_then) = &hist[u - 1];
    let wo = f * lam * w_then;
    let cut = |now: f64, then: f64| (w_now * now - wo * then) / (w_now - wo);
    let pair = win.pair(0, 1);
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-10 * b.abs().max(1.0);
    for li in 0..2 {
        let yy = cut(yy_now[li], yy_then[li]) / pair.var_y;
        let xx = cut(xx_now[li], xx_then[li]) / pair.var_x;
        assert!(
            close(pair.lagcorr_yy[li], yy),
            "lag {li}: {} against {yy}",
            pair.lagcorr_yy[li]
        );
        assert!(
            close(pair.lagcorr_xx[li], xx),
            "lag {li}: {} against {xx}",
            pair.lagcorr_xx[li]
        );
    }
    let xy = cut(*xy_now, *xy_then) / (pair.var_x.sqrt() * pair.var_y.sqrt());
    assert!(
        close(pair.lagcorr_xy[0], xy),
        "{} against {xy}",
        pair.lagcorr_xy[0]
    );
    // The window truncated: the whole history's reads otherwise.
    let whole = live.pair(0, 1);
    assert!((whole.lagcorr_yy[0] - pair.lagcorr_yy[0]).abs() > 1e-4);
}

/// What the lag moments add to a window's snapshot (docs/PLAN.md task
/// 137): `L·T + (L + 2C)·p·T` doubles and the width, as `window_lags`
/// documents.
#[test]
fn the_lag_moments_add_what_window_lags_says() {
    for (p, t, lags, cross) in [
        (3usize, 1usize, vec![1usize], None),
        (10, 2, vec![1, 2, 5], None),
        (10, 2, vec![1, 2, 5], Some(vec![])),
        (50, 1, vec![1, 5], Some(vec![1])),
    ] {
        let mut c = cfg(p, t);
        c.lags = lags.clone();
        c.cross_lags = cross.clone();
        let m = Marginal::new(c).unwrap();
        let got = crate::Footprint::footprint(&m.lag.as_ref().unwrap().moments());
        let (l, cl) = (lags.len(), cross.as_ref().map_or(lags.len(), Vec::len));
        let want = 8 * (l * t + (l + 2 * cl) * p * t) + std::mem::size_of::<usize>();
        assert_eq!(got, want, "p = {p}, T = {t}, L = {l}, C = {cl}");
    }
}

#[test]
fn matches_the_longhand_moments_with_weights_gaps_and_a_missing_target() {
    // Two targets, the second missing on every third row, uneven weights
    // (some zero), and clock gaps: each target's pair moments must equal
    // the decayed weighted sums over the rows where that target was
    // present, with the decay of every row -- present or not -- applied.
    let mut m = Marginal::new(cfg(3, 2)).unwrap();
    let mut s = 7u64;
    let mut long: Vec<Vec<Longhand>> = (0..2)
        .map(|_| {
            (0..3)
                .map(|_| Longhand {
                    xs: vec![],
                    ys: vec![],
                    ws: vec![],
                    lams: vec![],
                })
                .collect()
        })
        .collect();
    for i in 0..300 {
        let x: Vec<f64> = (0..3).map(|_| 2.0 * lcg(&mut s) + 100.0).collect();
        let y0 = x[0] - 0.5 * x[1] + 0.3 * lcg(&mut s);
        let y1 = -x[2] + 0.1 * lcg(&mut s);
        let y = [Some(y0), (i % 3 != 2).then_some(y1)];
        let w = match i % 7 {
            0 => 0.0,
            1 => 2.5,
            _ => 1.0,
        };
        let d = if i == 0 {
            0.0
        } else if i % 50 == 0 {
            15.0
        } else {
            1.0
        };
        let lam = m.cfg().decay.factor(d);
        for (t, yt) in y.iter().enumerate() {
            for j in 0..3 {
                let l = &mut long[t][j];
                // Every row ages the target's history; only a present
                // target adds a row.
                // A missing target is a zero-weight row in the longhand:
                // it adds nothing to the sums and `weights()` still
                // applies its decay factor to every earlier row.
                let (xv, yv, wv) = match yt {
                    Some(yt) => (x[j], *yt, w),
                    None => (0.0, 0.0, 0.0),
                };
                l.xs.push(xv);
                l.ys.push(yv);
                l.ws.push(wv);
                l.lams.push(lam);
            }
        }
        let step = m.step(&x, &y, d, w);
        assert!(step.pred.is_empty());
        assert!(step.n_eff.is_finite());
    }
    for (t, row) in long.iter().enumerate() {
        for (j, l) in row.iter().enumerate() {
            let (sw, sq, mx, my, sxx, syy, sxy) = l.moments();
            let p = m.pair(t, j);
            assert!(close(p.n_eff, sw, 1e-12), "W_{t}: {} vs {sw}", p.n_eff);
            assert!(
                close(p.n_kish, sw * sw / sq, 1e-12),
                "kish_{t}: {} vs {}",
                p.n_kish,
                sw * sw / sq
            );
            assert!(
                close(p.mean_x, mx, 1e-12),
                "mx[{t},{j}] {} vs {mx}",
                p.mean_x
            );
            assert!(close(p.mean_y, my, 1e-12), "my[{t}] {} vs {my}", p.mean_y);
            // Centred second moments around an offset of 100: the
            // Welford form keeps them to ~1e-12 relative; a raw
            // `E[x²] − m²` would have lost them.
            assert!(
                close(p.var_x, sxx, 1e-9),
                "sxx[{t},{j}] {} vs {sxx}",
                p.var_x
            );
            assert!(close(p.var_y, syy, 1e-9), "syy[{t}] {} vs {syy}", p.var_y);
            assert!(close(p.cov, sxy, 1e-9), "sxy[{t},{j}] {} vs {sxy}", p.cov);
            let corr = sxy / (sxx * syy).sqrt();
            assert!(
                close(p.corr, corr, 1e-9),
                "corr[{t},{j}] {} vs {corr}",
                p.corr
            );
            assert!(close(p.beta, sxy / sxx, 1e-9), "beta[{t},{j}]");
            let n = sw * sw / sq;
            let tt = corr * ((n - 2.0) / (1.0 - corr * corr)).sqrt();
            assert!(close(p.t, tt, 1e-9), "t[{t},{j}] {} vs {tt}", p.t);
        }
    }
    // The model-level weight counts every row, including the ones where
    // a target was missing: it is `W_0` here, since target 0 was always
    // present.
    assert_eq!(m.n_eff().to_bits(), m.target_weight(0).to_bits());
    assert!(m.target_weight(1) < m.target_weight(0));
}

#[test]
fn a_pair_is_the_ew_cov_of_the_two_columns_to_the_bit() {
    // `ew_cov` over `[x_j, y]` and the pair `(j, 0)` here, fed the same
    // rows: the same recursion in the same order gives the same bits --
    // moments, and the correlation.
    let mut m = Marginal::new(cfg(2, 1)).unwrap();
    let mut covs = [EwCov::new(2), EwCov::new(2)];
    let mut full = EwCovModel::new(EwCovCfg {
        n_features: 2,
        decay: Decay::Halflife(20.0),
        stats: vec![EwCovStat::Corr],
        min_weight: 0.0,
        precision_prior: None,
        mahal_quantiles: Vec::new(),
        pca: 0,
        pca_every: 1.0,
        max_rows_between_pca: u32::MAX,
        lags: Vec::new(),
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap();
    let mut s = 99u64;
    for i in 0..200 {
        let x = [lcg(&mut s) * 3.0, lcg(&mut s) + 5.0];
        let y = x[0] + 0.7 * x[1] + 0.2 * lcg(&mut s);
        let w = if i % 5 == 0 {
            0.0
        } else {
            1.0 + 0.5 * (i % 4) as f64
        };
        let d = if i == 0 { 0.0 } else { 0.5 + (i % 3) as f64 };
        let lam = m.cfg().decay.factor(d);
        // The ew_cov reference is read *before* the row too.
        let corr_ref = full.step(&[x[0], y], &[], d, w).pred[0];
        let before = m.pair(0, 0);
        if i > 0 {
            assert_eq!(before.corr.to_bits(), corr_ref.to_bits(), "row {i}");
        }
        m.step(&x, &[Some(y)], d, w);
        for (j, c) in covs.iter_mut().enumerate() {
            c.update(&[x[j], y], lam, w);
        }
    }
    for (j, c) in covs.iter().enumerate() {
        let p = m.pair(0, j);
        assert_eq!(p.n_eff.to_bits(), c.n_eff().to_bits());
        assert_eq!(p.mean_x.to_bits(), c.mean(0).to_bits());
        assert_eq!(p.mean_y.to_bits(), c.mean(1).to_bits());
        assert_eq!(p.var_x.to_bits(), c.var(0).to_bits());
        assert_eq!(p.var_y.to_bits(), c.var(1).to_bits());
        assert_eq!(p.cov.to_bits(), c.cov(0, 1).to_bits());
    }
}

#[test]
fn n_eff_is_the_weight_before_the_row_and_min_periods_gates_the_derived_values() {
    let mut c = cfg(1, 1);
    c.min_weight = vec![3.0];
    let mut m = Marginal::new(c).unwrap();
    let lam = 0.5f64.powf(1.0 / 20.0);
    let mut expect = 0.0;
    for i in 0..6 {
        let x = [i as f64];
        let step = m.step(
            &x,
            &[Some(2.0 * i as f64 + 1.0)],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
        assert_eq!(step.n_eff, expect, "row {i}");
        expect = if i == 0 { 1.0 } else { lam * expect + 1.0 };
        let p = m.pair(0, 0);
        // The moments are there from the first row; the derived values
        // wait for the weight.
        assert!(p.mean_x.is_finite());
        if p.n_eff < 3.0 {
            assert!(
                p.corr.is_nan() && p.beta.is_nan() && p.t.is_nan(),
                "row {i}"
            );
        } else {
            assert!(
                (p.corr - 1.0).abs() < 1e-12,
                "y = 2x + 1: corr 1, got {}",
                p.corr
            );
            assert!((p.beta - 2.0).abs() < 1e-9, "beta 2, got {}", p.beta);
            // `corr` is 1 to rounding, so `1 − corr²` is tiny or zero
            // and t is enormous or +inf: either is the honest value.
            assert!(p.t > 1e3, "perfect fit: t huge, got {}", p.t);
        }
    }
}

#[test]
fn a_zero_weight_first_row_and_a_constant_column_stay_finite() {
    let mut m = Marginal::new(cfg(2, 1)).unwrap();
    // Weight 0 on the very first row: nothing to average, no 0/0.
    let s0 = m.step(&[1.0, 2.0], &[Some(3.0)], 0.0, 0.0);
    assert_eq!(s0.n_eff, 0.0);
    let p = m.pair(0, 0);
    assert_eq!(p.n_eff, 0.0);
    assert!(p.n_kish.is_nan(), "0/0 before any weight");
    assert_eq!(p.mean_x, 0.0);
    assert!(p.corr.is_nan());
    // Then a constant feature (column 1): var_x = 0 exactly, corr and
    // beta NaN rather than infinite, everything else finite.
    for i in 0..20 {
        let xi = (i as f64).sin();
        m.step(&[xi, 7.0], &[Some(2.0 * xi)], 1.0, 1.0);
    }
    let p = m.pair(0, 1);
    assert_eq!(p.var_x, 0.0);
    assert_eq!(p.cov, 0.0);
    assert!(p.corr.is_nan() && p.beta.is_nan());
    assert!(p.mean_y.is_finite() && p.var_y > 0.0);
    let p = m.pair(0, 0);
    assert!((p.corr - 1.0).abs() < 1e-12);
    assert!((p.beta - 2.0).abs() < 1e-9);
}

#[test]
fn kish_size_of_unit_weights_tends_to_one_plus_lam_over_one_minus_lam() {
    let mut m = Marginal::new(cfg(1, 1)).unwrap();
    let mut s = 3u64;
    for i in 0..5000 {
        let x = lcg(&mut s);
        m.step(
            &[x],
            &[Some(x + lcg(&mut s))],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    let lam = 0.5f64.powf(1.0 / 20.0);
    let p = m.pair(0, 0);
    assert!(close(p.n_eff, 1.0 / (1.0 - lam), 1e-9));
    assert!(
        close(p.n_kish, (1.0 + lam) / (1.0 - lam), 1e-9),
        "{}",
        p.n_kish
    );
    // Unequal weights lower it: a single heavy row dominates.
    let mut m2 = Marginal::new(cfg(1, 1)).unwrap();
    for i in 0..50 {
        let w = if i == 49 { 1000.0 } else { 1.0 };
        m2.step(
            &[i as f64],
            &[Some(i as f64)],
            if i == 0 { 0.0 } else { 1.0 },
            w,
        );
    }
    let p2 = m2.pair(0, 0);
    assert!(p2.n_kish < 1.1, "one row carries the weight: {}", p2.n_kish);
    assert!(p2.n_eff > 1000.0);
}

#[test]
fn state_round_trips_and_continues_identically() {
    let mut m = Marginal::new(cfg(3, 2)).unwrap();
    let mut s = 11u64;
    let row = |s: &mut u64| {
        let x: Vec<f64> = (0..3).map(|_| lcg(s)).collect();
        let y = [Some(x[0] + lcg(s)), (lcg(s) > 0.0).then(|| x[1] - lcg(s))];
        (x, y)
    };
    for i in 0..50 {
        let (x, y) = row(&mut s);
        m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let bytes = rmp_serde::to_vec_named(&m.state()).unwrap();
    let mut r = Marginal::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    assert_eq!(r, m);
    for _ in 0..50 {
        let (x, y) = row(&mut s);
        let a = m.step(&x, &y, 1.0, 1.5);
        let b = r.step(&x, &y, 1.0, 1.5);
        assert_eq!(a, b);
    }
    assert_eq!(r, m);
    let wrong = Marginal::restore(
        &EwCovModel::new(EwCovCfg {
            n_features: 1,
            decay: Decay::Halflife(1.0),
            stats: vec![],
            min_weight: 0.0,
            precision_prior: None,
            mahal_quantiles: vec![],
            pca: 0,
            pca_every: 1.0,
            max_rows_between_pca: u32::MAX,
            lags: Vec::new(),
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
        })
        .unwrap()
        .state(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        wrong.contains("marginal") && wrong.contains("ew_cov"),
        "{wrong}"
    );
}

#[test]
fn shape_accessors() {
    let m = Marginal::new(cfg(4, 3)).unwrap();
    assert_eq!(m.n_features(), 4);
    assert_eq!(m.n_targets(), 3);
    assert_eq!(m.n_outputs(), 0);
    assert_eq!(m.n_eff(), 0.0);
    assert_eq!(m.state().model.kind(), "marginal");
    let p = m.predict(&[0.0; 4], 1.0);
    assert!(p.pred.is_empty() && p.n_eff == 0.0 && p.extra.is_none());
}
/// `d_clock` is the gap since the previous row, so the first row has none.
fn step_clock(i: usize) -> f64 {
    if i == 0 { 0.0 } else { 1.0 }
}

fn bins_cfg(n_bins: usize, warm_rows: usize) -> Box<crate::BinCfg> {
    Box::new(crate::BinCfg {
        n_bins,
        edges: None,
        rule: crate::BinRule::Quantile,
        warm_rows,
        budget_mib: None,
    })
}

fn marginal_with(bins: Option<Box<crate::BinCfg>>) -> Marginal {
    Marginal::new(MarginalCfg {
        n_features: 1,
        n_targets: 1,
        decay: Decay::Halflife(200.0),
        min_weight: vec![0.0],
        lags: Vec::new(),
        serial_rule: None,
        cross_lags: None,
        bins,
        feature_moments: FeatureMomentLayout::PerTarget,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap()
}

/// The warm-up rows are held, not spent. Learning the edges from the
/// first 50 rows and then replaying them must land on exactly the state
/// a model given those same edges up front reaches -- to the bit, since
/// the replay performs the same operations in the same order.
#[test]
fn learned_edges_lose_no_row() {
    let rows: Vec<(f64, f64)> = {
        let mut seed = 99u64;
        (0..400)
            .map(|_| {
                let x = lcg(&mut seed);
                (x, x.abs() + 0.2 * lcg(&mut seed))
            })
            .collect()
    };
    let mut learned = marginal_with(Some(bins_cfg(4, 50)));
    for (i, (x, y)) in rows.iter().enumerate() {
        OnlineModel::step(&mut learned, &[*x], &[Some(*y)], step_clock(i), 1.0);
    }
    let got = learned.pair(0, 0);
    assert_eq!(got.bin_edges.len(), 3, "four bins means three edges");

    let mut given = marginal_with(Some(Box::new(crate::BinCfg {
        n_bins: 4,
        edges: Some(vec![got.bin_edges.clone()]),
        rule: crate::BinRule::Quantile,
        warm_rows: 50,
        budget_mib: None,
    })));
    for (i, (x, y)) in rows.iter().enumerate() {
        OnlineModel::step(&mut given, &[*x], &[Some(*y)], step_clock(i), 1.0);
    }
    let want = given.pair(0, 0);
    assert_eq!(got.bin_n, want.bin_n, "bin weights");
    assert_eq!(got.bin_mean_y, want.bin_mean_y, "bin means");
    assert_eq!(got.bin_var_y, want.bin_var_y, "bin variances");
    assert_eq!(got.split_gain, want.split_gain);
    assert_eq!(got.split_at, want.split_at);
}

/// The same across targets, through the row update that bins each
/// feature once (E71, docs/PLAN.md task 122): five targets, each absent
/// on a random third of the rows, and three features, one of them
/// constant so that it keeps no edges. The replay lands on the state
/// given edges reach, pair by pair, every bin and split number to the
/// bit.
///
/// Except by a rounding where rows of weight zero fall inside the
/// warm-up: each is held as its decay alone, folded into the next held
/// row so that a run of them cannot grow the hold, and the replay ages
/// the histogram by the product, `scale·(a·b)` where given edges age it
/// `(scale·a)·b` (docs/MARGINAL-LAGS-AND-BINS.md, "Invariance"). Measured
/// here, the two differ by 1.5e-15 of the data's scale; the bound below
/// is ten times that, and the test fails if they stop differing, since
/// the docs would then understate the replay.
#[test]
fn learned_edges_lose_no_row_across_targets() {
    let (p, n_targets) = (3, 5);
    let mk = |edges: Option<Vec<Vec<f64>>>| {
        Marginal::new(MarginalCfg {
            n_features: p,
            n_targets,
            decay: Decay::Halflife(200.0),
            min_weight: vec![0.0; n_targets],
            lags: Vec::new(),
            serial_rule: None,
            cross_lags: None,
            bins: Some(Box::new(crate::BinCfg {
                n_bins: 6,
                edges,
                rule: crate::BinRule::Quantile,
                warm_rows: 60,
                budget_mib: None,
            })),
            feature_moments: FeatureMomentLayout::PerTarget,
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
        })
        .unwrap()
    };
    // The distance between two numbers on the data's scale, which is
    // about 1: equal bits (a NaN for a bin or a split that has none) are
    // 0 apart, and a NaN against a number is infinitely far.
    let apart = |a: f64, b: f64| -> f64 {
        if a.to_bits() == b.to_bits() {
            0.0
        } else if a.is_finite() && b.is_finite() {
            (a - b).abs() / a.abs().max(b.abs()).max(1.0)
        } else {
            f64::INFINITY
        }
    };
    let mut worst_with_zero_weights = 0.0_f64;
    for zero_weights in [false, true] {
        let mut seed = 31u64;
        let rows: Vec<(Vec<f64>, Vec<Option<f64>>, f64)> = (0..500)
            .map(|i| {
                let x = vec![lcg(&mut seed), 0.25, lcg(&mut seed).powi(3)];
                let y = (0..n_targets)
                    .map(|t| {
                        let v = x[0] * t as f64 + x[2].abs() + 0.1 * lcg(&mut seed);
                        (lcg(&mut seed) >= -1.0 / 3.0).then_some(v)
                    })
                    .collect();
                let w = if zero_weights && i % 17 == 5 {
                    0.0
                } else {
                    1.0
                };
                (x, y, w)
            })
            .collect();
        let run = |m: &mut Marginal| {
            for (i, (x, y, w)) in rows.iter().enumerate() {
                OnlineModel::step(m, x, y, step_clock(i), *w);
            }
        };
        let mut learned = mk(None);
        run(&mut learned);
        let edges: Vec<Vec<f64>> = (0..p).map(|j| learned.pair(0, j).bin_edges).collect();
        assert_eq!(edges[0].len(), 5, "six bins means five edges");
        assert!(edges[1].is_empty(), "a constant feature keeps no edges");
        let mut given = mk(Some(edges));
        run(&mut given);
        for t in 0..n_targets {
            for j in 0..p {
                let (a, b) = (learned.pair(t, j), given.pair(t, j));
                assert!(a.bin_n.iter().any(|n| *n > 0.0), "target {t} feature {j}");
                let numbers = |q: &Pair| {
                    let mut v =
                        [q.bin_n.clone(), q.bin_mean_y.clone(), q.bin_var_y.clone()].concat();
                    v.extend([q.split_gain, q.split_at, q.split_gain_t]);
                    v
                };
                let (na, nb) = (numbers(&a), numbers(&b));
                assert_eq!(na.len(), nb.len());
                let worst = na
                    .iter()
                    .zip(&nb)
                    .map(|(x, y)| apart(*x, *y))
                    .fold(0.0, f64::max);
                if zero_weights {
                    worst_with_zero_weights = worst_with_zero_weights.max(worst);
                } else {
                    assert_eq!(worst, 0.0, "target {t} feature {j}: {na:?} vs {nb:?}");
                }
            }
        }
    }
    assert!(
        worst_with_zero_weights > 0.0 && worst_with_zero_weights < 1.5e-14,
        "{worst_with_zero_weights:e} apart with rows of weight zero in the warm-up"
    );
}

/// A zero-weight row inside the warm-up teaches nothing: its feature
/// values cannot reach the histogram, so putting absurd ones there must
/// change nothing at all. (That it also advances the clock is the shared
/// rule, checked for every model in `model_contract.rs`.)
#[test]
fn a_zero_weight_row_in_the_warm_up_teaches_nothing() {
    let run = |junk: f64| {
        let mut m = marginal_with(Some(bins_cfg(4, 20)));
        let mut seed = 5u64;
        for i in 0..200 {
            let x = lcg(&mut seed);
            if i == 5 {
                OnlineModel::step(&mut m, &[junk], &[Some(junk)], 1.0, 0.0);
            }
            OnlineModel::step(&mut m, &[x], &[Some(x.abs())], step_clock(i), 1.0);
        }
        m.pair(0, 0)
    };
    let tame = run(0.5);
    let absurd = run(1e9);
    assert_eq!(
        tame.bin_edges, absurd.bin_edges,
        "a zero-weight row set an edge"
    );
    assert_eq!(
        tame.bin_n, absurd.bin_n,
        "a zero-weight row landed in a bin"
    );
    assert_eq!(tame.bin_mean_y, absurd.bin_mean_y);
    assert_eq!(tame.split_gain, absurd.split_gain);
}

/// What bins are for: a V-shaped relation has no linear signal at all,
/// and a split finds it. Checked against the gain computed the long way
/// over the same edges.
#[test]
fn a_v_shape_is_invisible_to_corr_and_obvious_to_a_split() {
    let mut m = marginal_with(Some(bins_cfg(8, 200)));
    let mut seed = 21u64;
    let mut rows = Vec::new();
    for i in 0..4000 {
        let x = lcg(&mut seed);
        let y = x.abs();
        OnlineModel::step(&mut m, &[x], &[Some(y)], step_clock(i), 1.0);
        rows.push((x, y));
    }
    let p = m.pair(0, 0);
    // The half-life is 200, so the effective sample is about 290 rows and
    // `corr` has a standard error near 0.06 whatever the truth is. The
    // claim is not that it is zero, but that it is noise: the split
    // explains an order of magnitude more of the target's variance than
    // the linear fit's own R-squared does.
    assert!(
        p.corr.abs() < 0.2,
        "a symmetric V should have no linear signal, got corr = {}",
        p.corr
    );
    assert!(
        p.split_gain > 10.0 * p.corr * p.corr,
        "gain {} against a linear R-squared of {}",
        p.split_gain,
        p.corr * p.corr
    );
    assert!(
        p.split_gain > 0.2,
        "a split should see it, got gain = {}",
        p.split_gain
    );
    // Overwhelming for an effective sample of ~290, even allowing for
    // the cut having been chosen by maximizing over seven candidates.
    assert!(
        p.split_gain_t > 10.0,
        "and say so loudly, got {}",
        p.split_gain_t
    );

    // The same gain, computed from the raw rows over the same edges.
    // Undecayed, so only the recent-weighted answer differs, and the
    // half-life is long enough here for that to be small.
    let edges = &p.bin_edges;
    let var_of = |rs: &[(f64, f64)]| {
        let n = rs.len() as f64;
        let m = rs.iter().map(|r| r.1).sum::<f64>() / n;
        rs.iter().map(|r| (r.1 - m) * (r.1 - m)).sum::<f64>() / n
    };
    let total = var_of(&rows);
    let best = edges
        .iter()
        .map(|c| {
            let (l, r): (Vec<_>, Vec<_>) = rows.iter().partition(|(x, _)| x < c);
            let (nl, nr) = (l.len() as f64, r.len() as f64);
            (total - (nl * var_of(&l) + nr * var_of(&r)) / (nl + nr)) / total
        })
        .fold(0.0_f64, f64::max);
    assert!(
        (p.split_gain - best).abs() < 0.05,
        "gain {} vs the long way {best}",
        p.split_gain
    );
}

/// Nothing is reported before the edges exist, and `bins` is refused
/// where it cannot be honoured.
#[test]
fn bins_before_and_beyond_what_it_can_do() {
    let mut m = marginal_with(Some(bins_cfg(4, 100)));
    for i in 0..10 {
        OnlineModel::step(&mut m, &[i as f64], &[Some(1.0)], step_clock(i), 1.0);
    }
    let p = m.pair(0, 0);
    assert!(
        p.bin_edges.is_empty(),
        "no curve before the edges are fixed"
    );
    assert!(p.split_gain.is_nan());

    let with_window = |bins| {
        Marginal::new(MarginalCfg {
            n_features: 1,
            n_targets: 1,
            decay: Decay::Halflife(50.0),
            min_weight: vec![0.0],
            lags: Vec::new(),
            serial_rule: None,
            cross_lags: None,
            bins,
            feature_moments: FeatureMomentLayout::PerTarget,
            window: Some(100.0),
            window_every: None,
            max_rows_between_snapshots: None,
        })
    };
    let err = with_window(Some(bins_cfg(4, 100))).unwrap_err();
    assert!(err.contains("bins and window"), "{err}");
    assert!(with_window(None).is_ok());

    let too_few = Marginal::new(MarginalCfg {
        n_features: 1,
        n_targets: 1,
        decay: Decay::Halflife(50.0),
        min_weight: vec![0.0],
        lags: Vec::new(),
        serial_rule: None,
        cross_lags: None,
        bins: Some(bins_cfg(8, 4)),
        feature_moments: FeatureMomentLayout::PerTarget,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap_err();
    assert!(too_few.contains("bin_warm_rows"), "{too_few}");
}

/// A feature with two values gets one edge, and a constant one gets
/// none, rather than an error: real inputs contain both.
#[test]
fn degenerate_features_keep_the_bins_they_can_support() {
    let mut m = Marginal::new(MarginalCfg {
        n_features: 3,
        n_targets: 1,
        decay: Decay::Halflife(100.0),
        min_weight: vec![0.0],
        lags: Vec::new(),
        serial_rule: None,
        cross_lags: None,
        bins: Some(bins_cfg(4, 20)),
        feature_moments: FeatureMomentLayout::PerTarget,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap();
    let mut seed = 3u64;
    for i in 0..100 {
        let x = lcg(&mut seed);
        let binary = if x > 0.0 { 1.0 } else { 0.0 };
        OnlineModel::step(
            &mut m,
            &[x, binary, 7.0],
            &[Some(binary)],
            step_clock(i),
            1.0,
        );
    }
    assert_eq!(
        m.pair(0, 0).bin_edges.len(),
        3,
        "a spread feature: four bins"
    );
    assert_eq!(
        m.pair(0, 1).bin_edges.len(),
        1,
        "a binary feature: two bins"
    );
    let constant = m.pair(0, 2);
    assert!(constant.bin_edges.is_empty(), "a constant feature: one bin");
    assert!(constant.split_gain.is_nan(), "and no split to report");
}

/// Two edge lists are the same thing in two layouts: bins learned from
/// a warm-up, or given up front. This is the given kind.
fn given_edges(edges: Vec<f64>) -> Box<crate::BinCfg> {
    Box::new(crate::BinCfg {
        n_bins: 2,
        edges: Some(vec![edges]),
        rule: crate::BinRule::Quantile,
        warm_rows: 2,
        budget_mib: None,
    })
}

/// The lagged moments are normalized (`E_w`), and their decay rides on
/// the target's weight through `a = lam·W/W'` on the rows that learn. A
/// row where the target is absent must therefore leave them exactly
/// where they are, as it leaves the pair moments: ageing them on their
/// own biased every lagged autocorrelation toward zero by the missing
/// fraction (`marglag` module doc).
#[test]
fn lag_moments_hold_where_the_target_is_missing() {
    let mut c = cfg(1, 1);
    c.lags = vec![1, 2];
    c.serial_rule = Some(SerialRule::Truncated);
    let mut m = Marginal::new(c).unwrap();
    let mut seed = 5u64;
    let mut x = 0.0;
    for i in 0..300 {
        x = 0.9 * x + lcg(&mut seed);
        let y = x + 0.1 * lcg(&mut seed);
        OnlineModel::step(&mut m, &[x], &[Some(y)], step_clock(i), 1.0);
    }
    let before = m.pair(0, 0);
    assert!(before.lagcorr_xx[0] > 0.5, "{:?}", before.lagcorr_xx);
    for _ in 0..100 {
        x = 0.9 * x + lcg(&mut seed);
        OnlineModel::step(&mut m, &[x], &[None], 1.0, 1.0);
    }
    let after = m.pair(0, 0);
    assert_eq!(before.var_x, after.var_x, "the pair moments hold");
    assert_eq!(before.corr, after.corr);
    assert_eq!(
        before.lagcorr_xx, after.lagcorr_xx,
        "so the lag moments hold"
    );
    assert_eq!(before.lagcorr_yy, after.lagcorr_yy);
    assert_eq!(before.lagcorr_xy, after.lagcorr_xy);
    assert_eq!(before.lagcorr_yx, after.lagcorr_yx);
    // `n_kish = W²/Q` is scale-free, so the correction is unchanged too
    // (to rounding: `W` and `Q` aged by `lam` and `lam²` a hundred times).
    assert!(close(before.n_serial, after.n_serial, 1e-12));
    assert!(after.n_eff < before.n_eff / 20.0, "only the weight aged");
}

/// A clock gap long enough that `lam` underflows to exactly zero, on a
/// row that carries no weight: the model's `n_eff` is zero after it, and
/// so must every target's be. The recursion `W' = lam·W + w` gives that
/// by itself; the `0/0` guard on `a` and `b` must not skip it.
#[test]
fn a_zero_weight_row_across_a_total_gap_ages_the_target_weight() {
    let mut c = cfg(1, 1);
    c.decay = Decay::Halflife(1.0);
    let mut m = Marginal::new(c).unwrap();
    for i in 0..50 {
        OnlineModel::step(&mut m, &[1.0], &[Some(1.0)], step_clock(i), 1.0);
    }
    assert!(m.pair(0, 0).n_eff > 1.9);
    assert_eq!(Decay::Halflife(1.0).factor(2000.0), 0.0, "a total gap");
    OnlineModel::step(&mut m, &[1.0], &[Some(1.0)], 2000.0, 0.0);
    assert_eq!(m.n_eff(), 0.0);
    let p = m.pair(0, 0);
    assert_eq!(p.n_eff, 0.0, "the target's weight decayed with the model's");
    assert!(p.n_kish.is_nan(), "0/0, reported as nothing");
    // And the stream resumes from nothing, as after a first row.
    OnlineModel::step(&mut m, &[2.0], &[Some(3.0)], 1.0, 1.0);
    let p = m.pair(0, 0);
    assert_eq!((p.n_eff, p.mean_x, p.mean_y), (1.0, 2.0, 3.0));
}

/// The histogram is the pair moments' companion, so a gap that takes the
/// pair's weight to nothing (`lam = 0`) must empty it too -- and a chain
/// of gaps whose product underflows the scale must not leave it dead.
/// The invariant: the bin weights sum to the target's `n_eff`.
#[test]
fn a_total_gap_empties_the_histogram_with_the_moments() {
    let sum = |p: &Pair| p.bin_n.iter().sum::<f64>();
    let mut c = cfg(1, 1);
    c.decay = Decay::Halflife(1.0);
    c.bins = Some(given_edges(vec![0.0]));
    let mut m = Marginal::new(c).unwrap();
    for i in 0..100 {
        let x = if i % 2 == 0 { -1.0 } else { 1.0 };
        OnlineModel::step(&mut m, &[x], &[Some(10.0)], step_clock(i), 1.0);
    }
    let p = m.pair(0, 0);
    assert!(p.bin_n.iter().all(|n| *n > 0.5), "{:?}", p.bin_n);
    assert!((sum(&p) - p.n_eff).abs() < 1e-9);

    // lam = 0: the old rows weigh nothing, and the histogram says so.
    OnlineModel::step(&mut m, &[-1.0], &[Some(0.0)], 2000.0, 1.0);
    let p = m.pair(0, 0);
    assert_eq!(p.bin_n, vec![1.0, 0.0], "one row, in the left bin");
    assert_eq!(p.bin_mean_y[0], 0.0, "and nothing of the 10s survives");
    assert!(p.bin_mean_y[1].is_nan());
    assert!(p.split_gain.is_nan(), "one bin, no variance, no split");
    for _ in 0..9 {
        OnlineModel::step(&mut m, &[-1.0], &[Some(0.0)], 1.0, 1.0);
    }
    let p = m.pair(0, 0);
    assert!(
        (sum(&p) - p.n_eff).abs() < 1e-9,
        "{} vs {}",
        sum(&p),
        p.n_eff
    );
    assert_eq!(p.bin_n[1], 0.0);

    // Two gaps whose factors multiply below the smallest scale the
    // histogram keeps: it folds, and keeps counting.
    let mut c = cfg(1, 1);
    c.decay = Decay::Halflife(1.0);
    c.bins = Some(given_edges(vec![0.0]));
    let mut m = Marginal::new(c).unwrap();
    OnlineModel::step(&mut m, &[-1.0], &[Some(1.0)], 0.0, 1.0);
    OnlineModel::step(&mut m, &[1.0], &[Some(1.0)], 465.0, 1.0);
    let p = m.pair(0, 0);
    assert!(
        (sum(&p) - p.n_eff).abs() < 1e-9,
        "{} vs {}",
        sum(&p),
        p.n_eff
    );
    OnlineModel::step(&mut m, &[1.0], &[Some(1.0)], 700.0, 1.0);
    for _ in 0..50 {
        OnlineModel::step(&mut m, &[1.0], &[Some(2.0)], 1.0, 1.0);
    }
    let p = m.pair(0, 0);
    assert!(p.n_eff > 1.99, "{}", p.n_eff);
    assert!(
        (sum(&p) - p.n_eff).abs() < 1e-9,
        "{} vs {}",
        sum(&p),
        p.n_eff
    );
    assert_eq!(p.bin_n[0], 0.0, "the first row is below f64 range");
    assert!((p.bin_mean_y[1] - 2.0).abs() < 1e-9, "{}", p.bin_mean_y[1]);
    assert!(p.bin_var_y[1] < 1e-9, "{}", p.bin_var_y[1]);
}

/// A cut that explains everything: `split_gain` is exactly one, and its
/// statistic is `+inf`, as `t` is at `corr = ±1`.
#[test]
fn a_perfect_split_has_an_infinite_statistic() {
    let mut c = cfg(1, 1);
    c.decay = Decay::Halflife(f64::INFINITY);
    c.bins = Some(given_edges(vec![0.0]));
    let mut m = Marginal::new(c).unwrap();
    for i in 0..100 {
        let x = if i % 2 == 0 { -1.0 } else { 1.0 };
        let y = if x > 0.0 { 1.0 } else { 0.0 };
        OnlineModel::step(&mut m, &[x], &[Some(y)], step_clock(i), 1.0);
    }
    let p = m.pair(0, 0);
    assert_eq!(p.split_gain, 1.0);
    assert_eq!(p.split_at, 0.0);
    assert_eq!(p.split_gain_t, f64::INFINITY);
    // The same target is a perfect line in `x`, and `t` agrees.
    assert_eq!(p.corr, 1.0);
    assert_eq!(p.t, f64::INFINITY);
}

// -----------------------------------------------------------------
// Sharded steps (docs/PLAN.md task 126).
// -----------------------------------------------------------------

/// One row of a stream that exercises everything a held row carries:
/// three targets, two of them often absent, weights of zero, clock gaps
/// long enough to fold the bins' scale and one that empties everything,
/// and the lags' ring cleared now and then.
struct ShardRow {
    x: Vec<f64>,
    y: Vec<Option<f64>>,
    d: f64,
    w: f64,
    clear: bool,
}

fn shard_stream(n: usize, p: usize) -> Vec<ShardRow> {
    shard_rows(n, p, true)
}

/// [`shard_stream`]'s rows, with its features that are not finite where
/// `not_finite` says, and every feature and target finite otherwise.
fn shard_rows(n: usize, p: usize, not_finite: bool) -> Vec<ShardRow> {
    let mut seed = 2026u64;
    let mut level = vec![0.0_f64; p];
    (0..n)
        .map(|i| {
            for (j, v) in level.iter_mut().enumerate() {
                *v = 0.9 * *v + lcg(&mut seed) + if j == 2 { 1e6 } else { 0.0 } * 1e-6;
            }
            // Feature 2 sits at a level, feature 4 holds one value for
            // long runs: the cases the compensated means and the runs are
            // for.
            // Feature indices wrap for a row narrower than five.
            let mut x = level.clone();
            x[2 % p] += 1e6;
            if p > 4 {
                x[4] = if (i / 40) % 2 == 0 { 3.0 } else { x[4] };
            }
            // A feature that is not finite now and then, and a row with
            // none finite: the ring must not take such a row, and the
            // bins take no cell for the feature (review 2026-09-26).
            if not_finite && i % 89 < p {
                x[i % 89] = f64::NAN;
            }
            if not_finite && i % 500 == 3 {
                x.iter_mut().for_each(|v| *v = f64::NAN);
            }
            let y0 = x[0] - 0.5 * x[1 % p] + 0.3 * lcg(&mut seed);
            let mut y = vec![
                Some(y0),
                (lcg(&mut seed) > -0.4).then(|| 2.0 * x[1 % p] + lcg(&mut seed)),
                (lcg(&mut seed) > 0.0).then(|| (x[0] * x[3 % p]).abs() + 0.1 * lcg(&mut seed)),
            ];
            // A row with no target at all, and one right after the total
            // gap at 218, when the histogram has just been wiped.
            if i % 101 == 9 || i == 219 {
                y.iter_mut().for_each(|v| *v = None);
            }
            let d = match i {
                0 => 0.0,
                _ if i % 211 == 7 => 1e6,
                _ if i % 53 == 11 => 300.0,
                _ => 1.0,
            };
            // Row 218 is a total gap on a row of no weight: every target
            // ages to nothing with no row to learn from.
            let w = if i % 13 == 5 || i == 218 {
                0.0
            } else {
                0.5 + (lcg(&mut seed) + 1.0) / 2.0
            };
            ShardRow {
                x,
                y,
                d,
                w,
                // Cleared now and then, and on the row that fills a batch
                // of 256, so a clear meets the flush in one step.
                clear: i % 97 == 50 || i == 255,
            }
        })
        .collect()
}

/// Every configuration a held row must carry exactly.
fn shard_cfgs(p: usize) -> Vec<(&'static str, MarginalCfg)> {
    let base = || {
        let mut c = cfg(p, 3);
        c.decay = Decay::Halflife(15.0);
        c.min_weight = vec![3.0; 3];
        c
    };
    let with = |f: &dyn Fn(&mut MarginalCfg)| {
        let mut c = base();
        f(&mut c);
        c
    };
    let learned = |c: &mut MarginalCfg| {
        c.bins = Some(Box::new(crate::BinCfg {
            n_bins: 4,
            edges: None,
            rule: crate::BinRule::Quantile,
            warm_rows: 40,
            budget_mib: None,
        }))
    };
    vec![
        ("moments", base()),
        (
            "lags",
            with(&|c| {
                c.lags = vec![1, 3, 8];
                c.serial_rule = Some(SerialRule::Geometric);
            }),
        ),
        (
            "cross lags",
            with(&|c| {
                c.lags = vec![1, 3, 8];
                c.cross_lags = Some(vec![3]);
            }),
        ),
        (
            "no cross lags",
            with(&|c| {
                c.lags = vec![1, 2];
                c.cross_lags = Some(vec![]);
            }),
        ),
        ("learned bins", with(&learned)),
        (
            "given bins",
            with(&|c| {
                c.bins = Some(Box::new(crate::BinCfg {
                    n_bins: 3,
                    edges: Some((0..p).map(|j| vec![-0.5 + j as f64 * 0.1, 0.5]).collect()),
                    rule: crate::BinRule::Quantile,
                    warm_rows: 2,
                    budget_mib: None,
                }))
            }),
        ),
        // One block of feature moments, which each shard takes whole for
        // its range (docs/PLAN.md task 125).
        (
            "shared feature moments",
            with(&|c| c.feature_moments = FeatureMomentLayout::Shared),
        ),
        (
            "shared with lags",
            with(&|c| {
                c.feature_moments = FeatureMomentLayout::Shared;
                c.lags = vec![1, 3, 8];
                c.cross_lags = Some(vec![3]);
                c.serial_rule = Some(SerialRule::Geometric);
            }),
        ),
        // A window takes no bins (`MarginalCfg::validate`); it takes lags,
        // whose moments its snapshots then hold (docs/PLAN.md task 137).
        ("window", with(&|c| c.window = Some(40.0))),
        (
            "window with lags",
            with(&|c| {
                c.window = Some(40.0);
                c.lags = vec![1, 3];
                c.cross_lags = Some(vec![1]);
                c.serial_rule = Some(SerialRule::Bartlett);
            }),
        ),
        (
            "window every 3",
            with(&|c| {
                c.window = Some(25.0);
                c.max_rows_between_snapshots = Some(3);
            }),
        ),
        // On the clock, alone and with a row cap (docs/PLAN.md task 162):
        // `takes` must say when the clock is due for the flush before it.
        (
            "window every 2.5 clock units",
            with(&|c| {
                c.window = Some(25.0);
                c.window_every = Some(2.5);
            }),
        ),
        (
            "window every 2.5 clock units or 2 rows",
            with(&|c| {
                c.window = Some(25.0);
                c.window_every = Some(2.5);
                c.max_rows_between_snapshots = Some(2);
            }),
        ),
    ]
}

fn state_bytes(m: &Marginal) -> Vec<u8> {
    rmp_serde::to_vec(&m.state()).unwrap()
}

/// Each shard on a thread of its own.
fn threaded(shards: &mut [MarginalShard<'_>]) {
    std::thread::scope(|s| {
        for sh in shards.iter_mut() {
            s.spawn(move || sh.run());
        }
    });
}

/// The last shard first: the order a pool runs them in is its own.
fn reversed(shards: &mut [MarginalShard<'_>]) {
    shards.iter_mut().rev().for_each(MarginalShard::run);
}

/// The whole point of the split (docs/PLAN.md task 126): whatever the
/// shard count, whichever threads run the shards and in what order, and
/// wherever the held rows are flushed, a sharded model is the unsplit
/// one to the bit -- every row's `n_eff`, and every byte of the state,
/// which holds every pair, lag, bin, run and window snapshot. Past a
/// batch (256 rows), across window snapshots, bin folds, a gap that
/// empties everything and a cleared ring.
#[test]
fn a_sharded_step_is_the_unsplit_step_to_the_bit() {
    let p = 7;
    let rows = shard_stream(700, p);
    let runners: [(&str, &ShardRunner<'_>); 3] = [
        ("in order", &run_in_order),
        ("threads", &threaded),
        ("reversed", &reversed),
    ];
    for (name, c) in shard_cfgs(p) {
        let mut plain = Marginal::new(c.clone()).unwrap();
        let steps: Vec<Step> = rows
            .iter()
            .map(|r| {
                if r.clear {
                    OnlineModel::clear_lags(&mut plain);
                }
                OnlineModel::step(&mut plain, &r.x, &r.y, r.d, r.w)
            })
            .collect();
        let want = state_bytes(&plain);
        for count in [2, 3, p, 50] {
            for (how, run) in runners {
                let shards = Shards { count, run };
                let mut m = Marginal::new(c.clone()).unwrap();
                let mut held = 0;
                for (i, r) in rows.iter().enumerate() {
                    if r.clear {
                        OnlineModel::clear_lags(&mut m);
                    }
                    let got = m.step_sharded(&r.x, &r.y, r.d, r.w, &shards);
                    assert_eq!(
                        got.n_eff.to_bits(),
                        steps[i].n_eff.to_bits(),
                        "{name}, {count} shards {how}: n_eff at row {i}"
                    );
                    held = held.max(m.held_rows());
                    // A flush wherever the caller likes, and the state
                    // there is the unsplit one's there.
                    if i == 333 {
                        m.flush(&shards);
                        let mut part = Marginal::new(c.clone()).unwrap();
                        for r in &rows[..=i] {
                            if r.clear {
                                OnlineModel::clear_lags(&mut part);
                            }
                            OnlineModel::step(&mut part, &r.x, &r.y, r.d, r.w);
                        }
                        assert!(
                            state_bytes(&m) == state_bytes(&part),
                            "{name}, {count} shards {how}: the state at row {i} differs"
                        );
                    }
                }
                // A window snapshot every row flushes every row.
                let most = if matches!(name, "window" | "window with lags") {
                    1
                } else {
                    2
                };
                assert!(held >= most, "{name}: rows were held ({held})");
                m.flush(&shards);
                assert_eq!(m.held_rows(), 0);
                assert!(
                    state_bytes(&m) == want,
                    "{name}, {count} shards {how}: the state differs"
                );
            }
        }
    }
}

/// The stream above reaches what it is meant to: rows held past a
/// batch, window snapshots taken while rows were held, bins that fold
/// and empty, a ring cleared with rows held -- each counted, since a
/// comparison of two models that never met a case says nothing about
/// it.
#[test]
fn the_sharded_stream_reaches_every_case() {
    let p = 7;
    let rows = shard_stream(700, p);
    let shards = Shards {
        count: 3,
        run: &run_in_order,
    };
    let cfgs = shard_cfgs(p);
    let find = |name: &str| cfgs.iter().find(|(n, _)| *n == name).unwrap().1.clone();
    for name in ["window", "window every 3"] {
        let mut m = Marginal::new(find(name)).unwrap();
        let mut snapshots_while_held = 0;
        for r in &rows {
            let win = m.win.as_ref().unwrap();
            if m.held_rows() > 0 && win.snaps.takes(win.clock + r.d) {
                snapshots_while_held += 1;
            }
            m.step_sharded(&r.x, &r.y, r.d, r.w, &shards);
        }
        assert!(
            snapshots_while_held > 100,
            "{name}: snapshots with rows held: {snapshots_while_held}"
        );
    }
    let mut m = Marginal::new(find("learned bins")).unwrap();
    let (mut folds, mut empties, mut folds_while_held) = (0, 0, 0);
    for r in &rows {
        let lam = m.cfg.decay.factor(r.d);
        if let Some(h) = m.bins.as_ref().and_then(|b| b.hist.as_ref())
            && h.folds_at(lam)
        {
            folds += 1;
            empties += usize::from(lam == 0.0);
            folds_while_held += usize::from(m.held_rows() > 0);
        }
        m.step_sharded(&r.x, &r.y, r.d, r.w, &shards);
    }
    assert!(
        m.bins.as_ref().unwrap().hist.is_some(),
        "the edges were learned"
    );
    assert!(
        folds > 2 && empties > 1 && folds_while_held == folds,
        "folds {folds}, of them empties {empties}, with rows held {folds_while_held}"
    );
    // Without a window, the batch fills, and a clear meets held rows.
    let mut lagged = Marginal::new(find("lags")).unwrap();
    let (mut most_held, mut cleared_while_held) = (0, 0);
    for r in &rows {
        if r.clear && lagged.held_rows() > 0 {
            cleared_while_held += 1;
        }
        if r.clear {
            OnlineModel::clear_lags(&mut lagged);
        }
        lagged.step_sharded(&r.x, &r.y, r.d, r.w, &shards);
        most_held = most_held.max(lagged.held_rows());
    }
    assert_eq!(most_held, batch_rows(p) - 1, "a full batch is flushed");
    assert!(
        cleared_while_held > 2,
        "clears with rows held: {cleared_while_held}"
    );
    // The stream's other cases, counted rather than assumed.
    let no_finite = rows
        .iter()
        .filter(|r| r.x.iter().all(|v| v.is_nan()))
        .count();
    let some_nan = rows
        .iter()
        .filter(|r| r.x.iter().any(|v| v.is_nan()) && !r.x.iter().all(|v| v.is_nan()))
        .count();
    let no_target = rows
        .iter()
        .filter(|r| r.y.iter().all(Option::is_none))
        .count();
    let gap_of_no_weight = rows.iter().filter(|r| r.d == 1e6 && r.w == 0.0).count();
    assert!(
        no_finite >= 1 && some_nan > 5 && no_target > 5 && gap_of_no_weight >= 1,
        "{no_finite} {some_nan} {no_target} {gap_of_no_weight}"
    );
    assert!(
        rows[batch_rows(p) - 1].clear,
        "a clear on the row that fills a batch"
    );
}

/// A plain step after sharded ones learns the held rows first, and a
/// state saved after a flush under one count continues under another,
/// or none: the count is not in the state.
#[test]
fn the_shard_count_is_not_in_the_state() {
    let p = 7;
    let rows = shard_stream(500, p);
    let cfgs = shard_cfgs(p);
    let c = &cfgs.iter().find(|(n, _)| *n == "lags").unwrap().1;
    let mut plain = Marginal::new(c.clone()).unwrap();
    for r in &rows {
        if r.clear {
            OnlineModel::clear_lags(&mut plain);
        }
        OnlineModel::step(&mut plain, &r.x, &r.y, r.d, r.w);
    }
    let five = Shards {
        count: 5,
        run: &run_in_order,
    };
    let two = Shards {
        count: 2,
        run: &threaded,
    };
    let mut m = Marginal::new(c.clone()).unwrap();
    for (i, r) in rows.iter().enumerate() {
        if r.clear {
            OnlineModel::clear_lags(&mut m);
        }
        match i {
            // Rows held, then a plain step: it learns them first.
            0..150 => {
                m.step_sharded(&r.x, &r.y, r.d, r.w, &five);
            }
            150..160 => {
                OnlineModel::step(&mut m, &r.x, &r.y, r.d, r.w);
            }
            // Saved and restored with rows flushed, then another count.
            160 => {
                m.flush(&five);
                let bytes = rmp_serde::to_vec(&m.state()).unwrap();
                m = Marginal::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
                m.step_sharded(&r.x, &r.y, r.d, r.w, &two);
            }
            _ => {
                m.step_sharded(&r.x, &r.y, r.d, r.w, &two);
            }
        }
    }
    m.flush(&two);
    assert!(state_bytes(&m) == state_bytes(&plain));
}

/// One shard or fewer is the unsplit step, row by row: nothing is held.
#[test]
fn one_shard_holds_nothing() {
    let rows = shard_stream(50, 7);
    let mut m = Marginal::new(cfg(7, 3)).unwrap();
    for r in &rows {
        for count in [0, 1] {
            let shards = Shards {
                count,
                run: &run_in_order,
            };
            m.step_sharded(&r.x, &r.y, r.d, r.w, &shards);
            assert_eq!(m.held_rows(), 0);
        }
    }
}

/// Reading a pair or the state with rows held is a caller's bug, and
/// says so in every build rather than reporting pairs that miss those
/// rows, or saving a state that can never learn them (review 2026-09-26,
/// A3).
#[test]
#[should_panic(expected = "flushed before a pair is read")]
fn a_pair_read_with_rows_held_is_refused() {
    let mut m = Marginal::new(cfg(4, 1)).unwrap();
    let shards = Shards {
        count: 2,
        run: &run_in_order,
    };
    m.step_sharded(&[1.0, 2.0, 3.0, 4.0], &[Some(1.0)], 0.0, 1.0, &shards);
    let _ = m.pair(0, 0);
}

/// `"auto"` splits a row only when a flush has work for two shards,
/// never into more than twice the threads, and lags, cross terms and
/// bins add to the work (docs/PERFORMANCE.md §25).
#[test]
fn auto_splits_only_what_fills_two_shards() {
    let shape = |p: usize, t: usize, lags: Vec<usize>, cross: Option<Vec<usize>>, bins: bool| {
        let mut c = cfg(p, t);
        c.lags = lags;
        c.cross_lags = cross;
        c.bins = bins.then(|| bins_cfg(16, 1_000));
        c
    };
    let moments = |p, t| shape(p, t, vec![], None, false);
    // Measured not to pay: split, 1,000 features ran 1.15 times as
    // fast at best and slower past two shards.
    assert_eq!(moments(1_000, 1).auto_shards(14), 1);
    assert_eq!(moments(4_000, 1).auto_shards(14), 6);
    assert_eq!(moments(10_000, 9).auto_shards(14), 28, "twice the threads");
    assert_eq!(moments(10_000, 9).auto_shards(1), 1, "one thread, no split");
    assert_eq!(moments(10_000, 9).auto_shards(0), 1);
    let lags = |cross| shape(1_000, 1, vec![1, 2, 5, 10, 20, 50], cross, false);
    assert!(lags(Some(vec![1])).auto_shards(14) > 1);
    assert!(lags(None).auto_shards(14) > lags(Some(vec![1])).auto_shards(14));
    assert!(
        shape(300, 1, vec![], None, true).auto_shards(14) > 1,
        "bins are work"
    );
    // The batch holds fewer rows of a wider row: the work per flush is
    // what counts.
    assert_eq!(batch_rows(10_000), 104);
    assert_eq!(batch_rows(1_000), BATCH_ROWS);
}

/// The warm-up hold's budget is what the held rows take, to the byte:
/// reserved at exactly the rows it will hold, again after a restore
/// partway through, with rows of weight zero taking nothing. And the
/// learned histogram that follows stays inside its budget
/// (docs/PLAN.md task 129).
#[test]
fn the_hold_budget_is_what_the_held_rows_take() {
    use crate::margbins::{HeldRow, histogram_bytes, hold_bytes};
    // Not a power of two, nor twice the restore point: a hold grown by
    // doubling cannot land on it by chance.
    let (p, t, warm, n_bins) = (9, 3, 37, 4);
    let held_bytes = |m: &Marginal| -> usize {
        let b = m.bins.as_ref().unwrap();
        b.held.capacity() * std::mem::size_of::<HeldRow>()
            + b.held
                .iter()
                .map(|r| {
                    r.x.capacity() * std::mem::size_of::<f64>()
                        + r.y.capacity() * std::mem::size_of::<Option<f64>>()
                })
                .sum::<usize>()
    };
    let mut c = cfg(p, t);
    c.bins = Some(bins_cfg(n_bins, warm));
    let mut m = Marginal::new(c).unwrap();
    let mut seed = 5u64;
    let (mut held, mut i, mut restored) = (0, 0, false);
    let row = |seed: &mut u64| -> (Vec<f64>, Vec<Option<f64>>) {
        let x = (0..p).map(|_| lcg(seed)).collect();
        let y = (0..t).map(|_| Some(lcg(seed))).collect();
        (x, y)
    };
    while held < warm - 1 {
        let (x, y) = row(&mut seed);
        let w = if i % 5 == 2 { 0.0 } else { 1.0 };
        held += usize::from(w > 0.0);
        OnlineModel::step(&mut m, &x, &y, step_clock(i), w);
        i += 1;
        if held == 20 && !restored {
            // A restored hold has no spare room; the next push reserves.
            let bytes = rmp_serde::to_vec(&m.state()).unwrap();
            m = Marginal::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
            restored = true;
        }
    }
    let b = m.bins.as_ref().unwrap();
    assert!(b.hist.is_none(), "the edges are not fixed yet");
    assert_eq!(b.held.len(), warm - 1);
    assert_eq!(
        b.held.capacity(),
        warm,
        "reserved at exactly the rows it holds"
    );
    // One row short of full: the hold's budget less that row's vectors.
    let last = p * std::mem::size_of::<f64>() + t * std::mem::size_of::<Option<f64>>();
    assert_eq!(held_bytes(&m) + last, hold_bytes(warm, p, t));
    // The next row fixes the edges and lets the hold go.
    let (x, y) = row(&mut seed);
    OnlineModel::step(&mut m, &x, &y, step_clock(i), 1.0);
    let b = m.bins.as_ref().unwrap();
    assert!(b.held.is_empty() && b.held.capacity() == 0);
    let hist = b.hist.as_ref().unwrap();
    assert!(hist.heap_bytes() <= histogram_bytes(p, t, p * n_bins));
}

/// A shard is sent to a pool's threads.
#[test]
fn a_shard_can_be_sent() {
    fn send<T: Send>() {}
    send::<MarginalShard<'_>>();
}

/// The window's snapshot counts every vector it holds in its footprint,
/// the per-target row counts included (docs/PLAN.md task 130).
#[test]
fn the_window_footprint_counts_every_vector() {
    let snap = |p: usize, t: usize| {
        let mut c = cfg(p, t);
        c.window = Some(10.0);
        let mut m = Marginal::new(c).unwrap();
        let mut s = 9u64;
        for i in 0..30 {
            let x: Vec<f64> = (0..p).map(|_| lcg(&mut s)).collect();
            let y: Vec<Option<f64>> = (0..t)
                .map(|k| (k == 0 || i % 3 != 1).then(|| lcg(&mut s)))
                .collect();
            m.step(&x, &y, step_clock(i), 1.0);
        }
        m.win.as_ref().unwrap().snaps.boundary().unwrap().1.clone()
    };
    crate::window::assert_footprint_counts_every_vector(&snap(2, 1), &snap(5, 3), "marginal");
}

#[test]
#[should_panic(expected = "flushed before the state is read")]
fn a_state_read_with_rows_held_is_refused() {
    let mut m = Marginal::new(cfg(4, 1)).unwrap();
    let shards = Shards {
        count: 2,
        run: &run_in_order,
    };
    m.step_sharded(&[1.0, 2.0, 3.0, 4.0], &[Some(1.0)], 0.0, 1.0, &shards);
    let _ = m.state();
}

/// `"auto"` counts what a windowed row flushes: a snapshot every row
/// flushes every row, which no width can keep busy, so it is not split;
/// a coarser cadence is, and one past the batch is bounded by the batch
/// (review 2026-09-26, A1).
#[test]
fn auto_does_not_split_a_window_snapshotted_every_row() {
    let mut c = cfg(10_000, 9);
    c.window = Some(40.0);
    assert_eq!(c.auto_shards(14), 1, "a snapshot every row");
    c.max_rows_between_snapshots = Some(64);
    assert!(c.auto_shards(14) > 1);
    c.max_rows_between_snapshots = Some(1_000);
    assert_eq!(
        c.auto_shards(14),
        cfg(10_000, 9).auto_shards(14),
        "the batch bounds it"
    );
    // A clock spacing bounds no count of rows, so alone it is the batch's;
    // at 0 it is every row (docs/PLAN.md task 162).
    c.max_rows_between_snapshots = None;
    c.window_every = Some(64.0);
    assert_eq!(c.auto_shards(14), cfg(10_000, 9).auto_shards(14));
    c.window_every = Some(0.0);
    assert_eq!(c.auto_shards(14), 1, "a snapshot every row");
}

/// `"auto"` at the Python suite's widths (`tests/test_marginal_shards.py`):
/// at 60 features and three targets it splits the bins alone, so the
/// other shapes' `"auto"` legs there compared the unsplit model with
/// itself (review 2026-09-26, F2); at 1,000 every shape splits on two
/// threads or more, the window at a snapshot every 128 rows; and the
/// 300-feature lag shape splits in four on two threads.
#[test]
fn auto_shards_at_the_python_suites_widths() {
    let shapes = |p: usize, every: usize| {
        let mut lags = cfg(p, 3);
        lags.lags = vec![1, 3, 7];
        lags.cross_lags = Some(vec![1]);
        let mut bins = cfg(p, 3);
        bins.bins = Some(bins_cfg(6, 50));
        let mut window = cfg(p, 3);
        window.window = Some(40.0);
        window.max_rows_between_snapshots = Some(every);
        [cfg(p, 3), lags, bins, window]
    };
    let at_60: Vec<usize> = shapes(60, 4).iter().map(|c| c.auto_shards(14)).collect();
    assert_eq!(at_60, [1, 1, 2, 1]);
    for threads in [2, 8, 14] {
        for (c, name) in shapes(1_000, 128)
            .iter()
            .zip(["moments", "lags", "bins", "window"])
        {
            let n = c.auto_shards(threads);
            assert!(
                n > 1 && n <= 2 * threads,
                "{name} on {threads} threads: {n}"
            );
        }
    }
    assert_eq!(
        shapes(300, 4)[1].auto_shards(2),
        4,
        "the child test's shape"
    );
}

/// A windowed model flushes before every snapshot: every row at the
/// default cadence, one row in eight at a row cap of 8, and at eight clock
/// units, a unit a row here (docs/PLAN.md task 162). Counted with a runner
/// of its own, so the regime is a fact and not an inference.
#[test]
fn a_window_snapshotted_every_row_flushes_every_row() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let rows = shard_stream(100, 3);
    for (every, rows_cap, at_least, at_most) in [
        (None, Some(1usize), 100, 100),
        (None, Some(8), 12, 14),
        (Some(8.0), None, 12, 14),
    ] {
        let flushes = AtomicUsize::new(0);
        let run = |s: &mut [MarginalShard<'_>]| {
            flushes.fetch_add(1, Ordering::Relaxed);
            run_in_order(s);
        };
        let shards = Shards {
            count: 2,
            run: &run,
        };
        let mut c = cfg(3, 3);
        c.window = Some(30.0);
        c.window_every = every;
        c.max_rows_between_snapshots = rows_cap;
        let mut m = Marginal::new(c).unwrap();
        for r in &rows {
            m.step_sharded(&r.x, &r.y, 1.0, r.w, &shards);
        }
        m.flush(&shards);
        let n = flushes.load(Ordering::Relaxed);
        assert!(
            (at_least..=at_most).contains(&n),
            "{every:?}, {rows_cap:?}: {n} flushes"
        );
    }
}

/// A short group holds only its rows: the hold grows by doubling up to
/// `bin_warm_rows`, never to it up front, so a bank of many short groups
/// does not pay every group's full hold (review 2026-09-26, A2). The
/// budget's count stays an upper bound, and a full hold still meets it
/// (`the_hold_budget_is_what_the_held_rows_take`).
#[test]
fn a_short_group_holds_only_its_rows() {
    let mut c = cfg(10, 1);
    c.bins = Some(bins_cfg(4, 10_000));
    let mut m = Marginal::new(c).unwrap();
    for i in 0..3 {
        m.step(&[i as f64; 10], &[Some(1.0)], step_clock(i), 1.0);
    }
    let held = &m.bins.as_ref().unwrap().held;
    assert_eq!(held.len(), 3);
    assert!(held.capacity() <= 8, "{}", held.capacity());
}

/// A histogram from a state written before the means' low parts, given
/// a row whose only feature is not finite: the plain and the sharded
/// path size the low parts alike, so the two states agree byte for
/// byte (review 2026-09-26, A7; the row itself is B1's).
#[test]
fn a_state_without_the_bins_low_parts_is_sized_the_same_by_both_paths() {
    let mut c = cfg(1, 1);
    c.bins = Some(given_edges(vec![0.0]));
    let mut m = Marginal::new(c).unwrap();
    m.step(&[0.5], &[Some(1.0)], 0.0, 1.0);
    let mut v = serde_json::to_value(m.state()).unwrap();
    crate::window::json_edit(&mut v, "mean_lo", &mut |x| *x = serde_json::json!([]));
    let old = Marginal::restore(&serde_json::from_value(v).unwrap()).unwrap();
    let (mut a, mut b) = (old.clone(), old);
    a.step(&[f64::NAN], &[Some(1.0)], 1.0, 1.0);
    let shards = Shards {
        count: 2,
        run: &run_in_order,
    };
    b.step_sharded(&[f64::NAN], &[Some(1.0)], 1.0, 1.0, &shards);
    b.flush(&shards);
    assert!(state_bytes(&a) == state_bytes(&b));
}

/// The sharded step's odd corners, each against the plain step: a
/// restore inside the bins' warm-up with the batch going on; a refusing
/// window budget tripped with rows held; the count changed between two
/// sharded steps with no flush between; one feature under two shards
/// (review 2026-09-26, A missing 5-8).
#[test]
fn the_sharded_steps_odd_corners_are_the_plain_step() {
    let p = 7;
    let rows = shard_stream(200, p);
    let cfgs = shard_cfgs(p);
    let find = |name: &str| cfgs.iter().find(|(n, _)| *n == name).unwrap().1.clone();
    let plain_run = |c: &MarginalCfg, upto: usize| {
        let mut m = Marginal::new(c.clone()).unwrap();
        for r in &rows[..upto] {
            if r.clear {
                OnlineModel::clear_lags(&mut m);
            }
            m.step(&r.x, &r.y, r.d, r.w);
        }
        m
    };
    let three = Shards {
        count: 3,
        run: &run_in_order,
    };
    let two = Shards {
        count: 2,
        run: &threaded,
    };
    // Restored at row 20 of a 40-row warm-up, sharded before and after.
    let c = find("learned bins");
    let mut m = Marginal::new(c.clone()).unwrap();
    for (i, r) in rows.iter().enumerate() {
        if i == 20 {
            m.flush(&three);
            let bytes = rmp_serde::to_vec(&m.state()).unwrap();
            m = Marginal::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        }
        if r.clear {
            OnlineModel::clear_lags(&mut m);
        }
        m.step_sharded(&r.x, &r.y, r.d, r.w, &three);
    }
    m.flush(&three);
    assert!(state_bytes(&m) == state_bytes(&plain_run(&c, rows.len())));
    // A refusing budget crossed with rows held: no snapshot forms after
    // it, so nothing flushes, and both models say they are over.
    let c = find("window every 3");
    let mut a = Marginal::new(c.clone()).unwrap();
    let mut b = Marginal::new(c.clone()).unwrap();
    let tiny = Some(crate::WindowBudget::Refuse(1e-9));
    a.set_window_budget(tiny);
    b.set_window_budget(tiny);
    for r in &rows {
        a.step(&r.x, &r.y, r.d, r.w);
        b.step_sharded(&r.x, &r.y, r.d, r.w, &three);
    }
    assert!(b.held_rows() > 100, "no flush once the budget refuses");
    b.flush(&three);
    assert!(a.window_over_budget().is_some());
    assert_eq!(a.window_over_budget(), b.window_over_budget());
    assert!(state_bytes(&a) == state_bytes(&b));
    // The count changes with no flush between.
    let c = find("lags");
    let mut m = Marginal::new(c.clone()).unwrap();
    for (i, r) in rows.iter().enumerate() {
        if r.clear {
            OnlineModel::clear_lags(&mut m);
        }
        m.step_sharded(&r.x, &r.y, r.d, r.w, if i < 100 { &three } else { &two });
    }
    m.flush(&two);
    assert!(state_bytes(&m) == state_bytes(&plain_run(&c, rows.len())));
    // One feature, two shards: held, and one shard at the flush.
    let narrow = shard_stream(50, 1);
    let mut c = cfg(1, 3);
    c.lags = vec![1, 2];
    let mut m = Marginal::new(c.clone()).unwrap();
    let mut plain = Marginal::new(c).unwrap();
    for r in &narrow {
        m.step_sharded(&r.x, &r.y, r.d, r.w, &two);
        plain.step(&r.x, &r.y, r.d, r.w);
    }
    assert!(m.held_rows() > 0);
    m.flush(&two);
    assert!(state_bytes(&m) == state_bytes(&plain));
}

fn shared(p: usize, t: usize) -> MarginalCfg {
    MarginalCfg {
        feature_moments: FeatureMomentLayout::Shared,
        ..cfg(p, t)
    }
}

/// Every number a pair reports, as bits: two pairs agree only where each
/// is the other's to the bit, NaN included.
fn pair_bits(m: &Marginal, t: usize, j: usize) -> Vec<u64> {
    let q = m.pair(t, j);
    [
        q.n_eff, q.n_kish, q.mean_x, q.var_x, q.mean_y, q.var_y, q.cov, q.corr, q.beta, q.t,
        q.n_serial, q.t_serial, q.phi_x, q.phi_y,
    ]
    .iter()
    .chain(&q.lagcorr_xx)
    .chain(&q.lagcorr_yy)
    .chain(&q.lagcorr_xy)
    .chain(&q.lagcorr_yx)
    .map(|v| v.to_bits())
    .collect()
}

/// A row of `p` features and `t` targets, each target loading on its own
/// feature, and the row's clock step and weight: a zero-weight first row,
/// a zero weight every eleventh, and a clock gap every thirteenth.
fn shared_row(s: &mut u64, i: usize, p: usize, t: usize) -> (Vec<f64>, Vec<f64>, f64, f64) {
    let x: Vec<f64> = (0..p).map(|_| 3.0 + lcg(s)).collect();
    let y: Vec<f64> = (0..t).map(|k| 0.5 * x[k % p] + lcg(s)).collect();
    let d = if i == 0 {
        0.0
    } else if i.is_multiple_of(13) {
        7.0
    } else {
        1.0
    };
    let w = if i == 0 || i % 11 == 4 {
        0.0
    } else {
        0.5 + lcg(s).abs()
    };
    (x, y, d, w)
}

/// docs/PLAN.md task 125: with every target on every learned row, each
/// target's mix is the row's and each target's feature means are the
/// shared ones, so `"shared"` reports `"per_target"`'s numbers to the bit
/// -- zero-weight rows, a zero-weight first row and clock gaps included --
/// from one block of feature moments where the default keeps one a target.
#[test]
fn shared_feature_moments_are_per_target_to_the_bit_where_every_target_is_present() {
    // One target runs the kernel's single pass, several its two; the
    // lags with every cross term, some and none.
    let plain = |_: &mut MarginalCfg| {};
    let every = |c: &mut MarginalCfg| {
        c.lags = vec![1, 2, 5];
        c.serial_rule = Some(SerialRule::Geometric);
    };
    let some = |c: &mut MarginalCfg| {
        c.lags = vec![1, 3];
        c.cross_lags = Some(vec![3]);
        c.serial_rule = Some(SerialRule::Bartlett);
    };
    let none = |c: &mut MarginalCfg| {
        c.lags = vec![1, 2];
        c.cross_lags = Some(vec![]);
    };
    let cases: [&dyn Fn(&mut MarginalCfg); 4] = [&plain, &every, &some, &none];
    for (p, t) in [(4, 1), (4, 3)] {
        for set in cases {
            shared_against_per_target(p, t, set);
        }
    }
}

fn shared_against_per_target(p: usize, t: usize, set: &dyn Fn(&mut MarginalCfg)) {
    let (mut a, mut b) = (cfg(p, t), shared(p, t));
    set(&mut a);
    set(&mut b);
    let mut per = Marginal::new(a).unwrap();
    let mut one = Marginal::new(b).unwrap();
    let mut s = 7u64;
    for i in 0..400 {
        let (x, y, d, w) = shared_row(&mut s, i, p, t);
        let y: Vec<Option<f64>> = y.into_iter().map(Some).collect();
        per.step(&x, &y, d, w);
        one.step(&x, &y, d, w);
        for (tt, j) in (0..t).flat_map(|tt| (0..p).map(move |j| (tt, j))) {
            assert_eq!(
                pair_bits(&per, tt, j),
                pair_bits(&one, tt, j),
                "T = {t}, row {i}, pair {tt},{j}"
            );
        }
    }
    assert_eq!((per.mx.len(), per.sxx.len()), (p * t, p * t));
    assert_eq!((one.mx.len(), one.sxx.len(), one.sxy.len()), (p, p, p * t));
}

/// Under the lags, with a target absent on every third row, the
/// feature's lagged autocorrelation is the feature's own over every
/// learned row: a `"per_target"` model whose one target is on every row
/// reads the same numbers, to the bit, since its target's mix is then
/// the row's.
#[test]
fn shared_lag_moments_are_the_features_over_every_learned_row() {
    let p = 3;
    let set = |c: &mut MarginalCfg| {
        c.lags = vec![1, 2, 4];
        c.cross_lags = Some(vec![1]);
    };
    let (mut a, mut b) = (shared(p, 2), cfg(p, 1));
    set(&mut a);
    set(&mut b);
    let mut one = Marginal::new(a).unwrap();
    let mut always = Marginal::new(b).unwrap();
    let mut s = 13u64;
    for i in 0..400 {
        let (x, y, d, w) = shared_row(&mut s, i, p, 2);
        let yy = [Some(y[0]), (i % 3 != 1).then_some(y[1])];
        one.step(&x, &yy, d, w);
        always.step(&x, &[Some(y[0])], d, w);
        for j in 0..p {
            let (q, r) = (one.pair(1, j), always.pair(0, j));
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&q.lagcorr_xx),
                bits(&r.lagcorr_xx),
                "row {i}, feature {j}"
            );
            assert_eq!(q.var_x.to_bits(), r.var_x.to_bits(), "row {i}, feature {j}");
        }
    }
}

/// The single pass under one target, with the target absent on every
/// third row: the feature moments move on every learned row, as the
/// column's own `EwCov`, to the bit, and the covariance only on the
/// target's rows.
#[test]
fn one_shared_target_absent_on_some_rows_keeps_the_features_over_every_row() {
    let p = 3;
    let mut one = Marginal::new(shared(p, 1)).unwrap();
    let mut cols: Vec<crate::EwCov> = (0..p).map(|_| crate::EwCov::new(1)).collect();
    let mut s = 5u64;
    let mut cov_before = vec![0.0f64; p];
    for i in 0..300 {
        let (x, y, d, w) = shared_row(&mut s, i, p, 1);
        let present = i % 3 != 1;
        let lam = Decay::Halflife(20.0).factor(d);
        for (j, c) in cols.iter_mut().enumerate() {
            c.update(&[x[j]], lam, w);
        }
        one.step(&x, &[present.then_some(y[0])], d, w);
        for (j, c) in cols.iter().enumerate() {
            let q = one.pair(0, j);
            assert_eq!(q.mean_x.to_bits(), c.mean(0).to_bits(), "row {i}");
            assert_eq!(q.var_x.to_bits(), c.var(0).max(0.0).to_bits(), "row {i}");
            if !present {
                assert_eq!(q.cov.to_bits(), cov_before[j].to_bits(), "row {i}");
            }
            cov_before[j] = q.cov;
        }
    }
}

/// Where a target is absent on some rows, `"shared"` is the other
/// estimator the docs name: its feature moments are the feature's over
/// every learned row -- an `EwCov` of the column alone, to the bit -- and
/// its covariance is the recursion the module states, stepped with the
/// target's mix and centred on that shared mean, written out here. The
/// target present on every row keeps `"per_target"`'s numbers to the
/// bit; the other's covariance moves off them.
#[test]
fn shared_feature_moments_are_the_features_over_every_learned_row() {
    let (p, t) = (3, 2);
    let mut per = Marginal::new(cfg(p, t)).unwrap();
    let mut one = Marginal::new(shared(p, t)).unwrap();
    let mut cols: Vec<crate::EwCov> = (0..p).map(|_| crate::EwCov::new(1)).collect();
    // The longhand: target 1's covariance with each feature, its mean
    // and weight, and the shared means before each row.
    let (mut c1, mut m1, mut w1) = (vec![0.0; p], 0.0f64, 0.0f64);
    let mut s = 11u64;
    let mut moved = false;
    for i in 0..600 {
        let (x, y, d, w) = shared_row(&mut s, i, p, t);
        let present = i % 3 != 1;
        let yy = vec![Some(y[0]), present.then_some(y[1])];
        let lam = Decay::Halflife(20.0).factor(d);
        let means: Vec<f64> = cols.iter().map(|c| c.mean(0)).collect();
        if present {
            let w_new = lam * w1 + w;
            if w_new > 0.0 {
                let (a, b) = (lam * w1 / w_new, w / w_new);
                let dy = y[1] - m1;
                for j in 0..p {
                    c1[j] = a * c1[j] + a * b * (x[j] - means[j]) * dy;
                }
                m1 += b * dy;
                w1 = w_new;
            } else {
                w1 = 0.0;
            }
        } else {
            w1 *= lam;
        }
        for (j, c) in cols.iter_mut().enumerate() {
            c.update(&[x[j]], lam, w);
        }
        per.step(&x, &yy, d, w);
        one.step(&x, &yy, d, w);
        for j in 0..p {
            assert_eq!(pair_bits(&per, 0, j), pair_bits(&one, 0, j), "row {i}");
            let q = one.pair(1, j);
            assert_eq!(q.mean_x.to_bits(), cols[j].mean(0).to_bits(), "row {i}");
            assert_eq!(
                q.var_x.to_bits(),
                cols[j].var(0).max(0.0).to_bits(),
                "row {i}"
            );
            assert!(
                (q.cov - c1[j]).abs() <= 1e-12 * c1[j].abs().max(1.0),
                "row {i}: {} vs {}",
                q.cov,
                c1[j]
            );
            moved |= q.cov != per.pair(1, j).cov;
        }
    }
    assert!(
        moved,
        "a target absent on some rows reads another covariance"
    );
}

/// `"shared"` takes no window; the refusal says why and what to use
/// instead. It takes lags.
#[test]
fn shared_feature_moments_refuse_a_window_by_name() {
    let windowed = MarginalCfg {
        window: Some(40.0),
        ..shared(2, 2)
    };
    let e = windowed.validate().unwrap_err();
    assert!(
        e.contains("\"shared\" takes no window") && e.contains("per_target"),
        "{e}"
    );
    let lagged = MarginalCfg {
        lags: vec![1],
        ..shared(2, 2)
    };
    lagged.validate().unwrap();
}

/// A shared state round-trips, and one whose feature moments are a pair's
/// width is refused, as any wrong shape is (review 2026-09-18, B3).
#[test]
fn a_shared_state_round_trips_and_a_wrong_width_is_refused() {
    let mut m = Marginal::new(shared(3, 2)).unwrap();
    let mut s = 3u64;
    for i in 0..50 {
        let (x, y, d, w) = shared_row(&mut s, i, 3, 2);
        let y: Vec<Option<f64>> = y.into_iter().map(Some).collect();
        m.step(&x, &y, d, w);
    }
    let back = Marginal::restore(&m.state()).unwrap();
    assert_eq!(state_bytes(&back), state_bytes(&m));
    let mut st = m.state();
    let crate::ModelState::Marginal(inner) = &mut st.model else {
        unreachable!()
    };
    inner.sxx = vec![0.0; 6];
    assert!(matches!(
        Marginal::restore(&st),
        Err(StateError::Invalid(e)) if e.contains("wrong shape")
    ));
}

// -----------------------------------------------------------------
// Task 158: the survivors of the weekly mutation pass.
// -----------------------------------------------------------------

/// PLAN §13.4 for the model's own weights: under a window, `n_eff` and
/// each target's weight are the weight of the rows inside it, computed
/// directly from those rows -- every row for `n_eff`, the target's own
/// for its weight -- and the step reports `n_eff`. The boundary is
/// inclusive, and rows do age out, which the test counts.
#[test]
fn the_models_weights_under_a_window_are_the_rows_inside_it() {
    let (half_life, window) = (25.0, 70.0);
    let mut c = cfg(1, 2);
    c.decay = Decay::Halflife(half_life);
    c.window = Some(window);
    let mut m = Marginal::new(c).unwrap();
    let mut seed = 5u64;
    let (mut t, mut ws, mut second, mut clock) = (vec![], vec![], vec![], 0.0);
    let mut aged_out = 0;
    for i in 0..140 {
        let d = if i == 0 {
            0.0
        } else {
            0.25 * (2.0 + (lcg(&mut seed).abs() * 8.0).floor())
        };
        clock += d;
        let x = lcg(&mut seed);
        let w = 0.5 + lcg(&mut seed).abs();
        let on = i % 3 != 0;
        let step = OnlineModel::step(&mut m, &[x], &[Some(x), on.then_some(-x)], d, w);
        t.push(clock);
        ws.push(w);
        second.push(on);
        let inside = |j: usize| clock - t[j] <= window;
        let decayed = |j: &usize| ws[*j] * 0.5_f64.powf((clock - t[*j]) / half_life);
        let all: f64 = (0..=i).filter(|&j| inside(j)).map(|j| decayed(&j)).sum();
        let own: f64 = (0..=i)
            .filter(|&j| inside(j) && second[j])
            .map(|j| decayed(&j))
            .sum();
        aged_out += usize::from((0..=i).any(|j| !inside(j)));
        assert!(
            close(m.n_eff(), all, 1e-9),
            "row {i}: {} vs {all}",
            m.n_eff()
        );
        assert!(close(m.target_weight(0), all, 1e-9), "row {i}: target 0");
        assert!(close(m.target_weight(1), own, 1e-9), "row {i}: target 1");
        let next = m.predict(&[0.0], 0.0);
        assert_eq!(next.n_eff, m.n_eff());
        if i > 0 {
            assert!(step.n_eff > 0.0);
        }
    }
    assert!(aged_out > 50, "rows aged out on {aged_out} reads");
}

/// A window left a crumb of its weight -- 1e-10 beside a row of 1e3
/// that has aged out -- is empty: `EMPTY_FRACTION` of the untruncated
/// weight is the line, at a weight above one as below it, and the
/// model's `n_eff`, the target's weight and the pair all say 0.
#[test]
fn a_window_left_a_crumb_of_its_weight_is_empty() {
    let mut c = cfg(1, 1);
    c.decay = Decay::Halflife(f64::INFINITY);
    c.window = Some(2.0);
    let mut m = Marginal::new(c).unwrap();
    OnlineModel::step(&mut m, &[1.0], &[Some(2.0)], 0.0, 1e3);
    OnlineModel::step(&mut m, &[3.0], &[Some(5.0)], 3.0, 1e-10);
    let left = m.w_sum - m.boundary().expect("the first row aged out").0.w_sum;
    assert!(left > 0.0 && left <= crate::window::EMPTY_FRACTION * m.w_sum);
    assert_eq!(m.n_eff(), 0.0);
    assert_eq!(m.target_weight(0), 0.0);
    let q = m.pair(0, 0);
    assert_eq!(q.n_eff, 0.0);
    assert!(q.mean_x.is_nan() && q.cov.is_nan());
}

/// A window's statistics are means, so the scale of the weights does
/// not reach them: the same stream at `2^-30` of the weights -- a scale
/// every product takes exactly -- reports the same pair and lag
/// correlations to the bit, and `n_eff` at that scale. The emptiness
/// line is a fraction of the weight and scales with it; drawn at a
/// fixed weight instead, it would empty every window of light rows.
#[test]
fn a_windows_statistics_do_not_depend_on_the_weights_scale() {
    let scale = 2f64.powi(-30);
    let run = |s: f64| {
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(25.0);
        c.window = Some(40.0);
        c.lags = vec![1, 2];
        let mut m = Marginal::new(c).unwrap();
        let mut seed = 13u64;
        let mut x = 0.0;
        for i in 0..120 {
            x = 0.7 * x + lcg(&mut seed);
            let y = 0.5 * x + lcg(&mut seed);
            let w = s * (0.5 + lcg(&mut seed).abs());
            OnlineModel::step(&mut m, &[x], &[Some(y)], step_clock(i), w);
        }
        m
    };
    let (a, b) = (run(1.0), run(scale));
    assert!(a.boundary().is_some(), "rows aged out of the window");
    assert!(b.n_eff() < 1e-6, "light rows: {}", b.n_eff());
    assert_eq!(b.n_eff(), a.n_eff() * scale);
    assert_eq!(b.target_weight(0), a.target_weight(0) * scale);
    let (p, q) = (a.pair(0, 0), b.pair(0, 0));
    assert_eq!(q.n_eff, p.n_eff * scale);
    assert!(
        p.lagcorr_xx
            .iter()
            .chain(&p.lagcorr_xy)
            .all(|v| v.is_finite())
    );
    let numbers = |v: &Pair| {
        [
            v.n_kish, v.mean_x, v.var_x, v.mean_y, v.var_y, v.cov, v.corr, v.beta, v.t,
        ]
        .into_iter()
        .chain(v.lagcorr_xx.iter().copied())
        .chain(v.lagcorr_yy.iter().copied())
        .chain(v.lagcorr_xy.iter().copied())
        .chain(v.lagcorr_yx.iter().copied())
        .map(f64::to_bits)
        .collect::<Vec<_>>()
    };
    assert_eq!(numbers(&p), numbers(&q));
}

/// `"geometric"` against its definition: `rho(l) = phi^l` fitted on
/// `log rho` through the origin over the lags where `rho > 0`, and the
/// factor `1 + 2·Σ_{l≥1} (phi_x·phi_y)^l`, here summed term by term. A
/// lag at zero is not one the fit can use, and one usable lag on a side
/// is no fit.
#[test]
fn the_geometric_factor_is_its_definition() {
    let (f, px, py) = serial_factor(SerialRule::Geometric, &[1, 2], &[0.5, 0.25], &[0.8, 0.64]);
    assert!(
        (px - 0.5).abs() < 1e-14 && (py - 0.8).abs() < 1e-14,
        "{px} {py}"
    );
    let tail: f64 = (1..2000).map(|l| (px * py).powi(l)).sum();
    assert!(
        (f - (1.0 + 2.0 * tail)).abs() < 1e-12,
        "{f} against {}",
        1.0 + 2.0 * tail
    );
    let (f, px0, _) = serial_factor(
        SerialRule::Geometric,
        &[1, 2, 3],
        &[0.5, 0.25, 0.0],
        &[0.8, 0.64, 0.512],
    );
    assert_eq!(px0, px, "the lag at zero left out");
    assert!(f.is_finite());
    let (f, px, py) = serial_factor(SerialRule::Geometric, &[1, 2], &[0.5, -0.1], &[0.8, 0.64]);
    assert!(f.is_nan() && px.is_nan(), "one usable lag: {f} {px}");
    assert!((py - 0.8).abs() < 1e-14);
}

/// A row of weight 0 takes no step in the shared feature means either,
/// to the bit, under one target (the kernel's single pass) and three
/// (its two): as in [`a_row_of_no_weight_leaves_the_means_as_they_were`],
/// the rows 0.7 and 5.292162135665459 at unit weight and no decay leave
/// each mean with a low part of a whole rounding step, where a zero step
/// would round the double afresh (`crate::comp::add`), and the model
/// given a zero-weight row there is the model without it.
#[test]
fn a_row_of_no_weight_leaves_the_shared_means_as_they_were() {
    for t in [1, 3] {
        let mut c = shared(2, t);
        c.decay = Decay::Halflife(f64::INFINITY);
        let mut once = Marginal::new(c).unwrap();
        for v in [0.7, 5.292162135665459] {
            OnlineModel::step(&mut once, &[v, -v], &vec![Some(v); t], 1.0, 1.0);
        }
        assert!(
            once.mx.iter().zip(&once.mx_lo).all(|(m, lo)| m + lo != *m),
            "{t} targets: a whole step below each double: {:?} {:?}",
            once.mx,
            once.mx_lo
        );
        let mut twice = once.clone();
        OnlineModel::step(&mut twice, &[1.0, -1.0], &vec![Some(1.0); t], 1.0, 0.0);
        assert_eq!((&once.mx, &once.mx_lo), (&twice.mx, &twice.mx_lo), "{t}");
        for k in 0..t {
            for j in 0..2 {
                assert_eq!(
                    pair_bits(&once, k, j),
                    pair_bits(&twice, k, j),
                    "({k}, {j})"
                );
            }
        }
    }
}

/// The bins' part of a restored state (review 2026-09-18, B3), one
/// corruption at a time: a held warm-up row short a feature, or with a
/// target too many, and a histogram of another width with nothing held.
#[test]
fn a_bins_state_of_another_shape_is_refused() {
    use crate::ModelState;
    let refused = |m: &Marginal, f: &dyn Fn(&mut Binned)| {
        let mut s = m.state();
        let ModelState::Marginal(inner) = &mut s.model else {
            unreachable!()
        };
        f(inner.bins.as_mut().unwrap());
        match Marginal::restore(&s) {
            Err(StateError::Invalid(e)) => e.contains("wrong shape"),
            Ok(_) => false,
            Err(e) => panic!("{e}"),
        }
    };
    let mut c = cfg(2, 1);
    c.bins = Some(bins_cfg(3, 50));
    let mut warm = Marginal::new(c).unwrap();
    for i in 0..5 {
        let v = i as f64;
        OnlineModel::step(&mut warm, &[v, -v], &[Some(v)], step_clock(i), 1.0);
    }
    assert_eq!(warm.bins.as_ref().unwrap().held.len(), 5);
    assert!(!refused(&warm, &|_| {}));
    assert!(
        refused(&warm, &|b| {
            b.held[1].x.pop();
        }),
        "a row short a feature"
    );
    assert!(
        refused(&warm, &|b| b.held[1].y.push(None)),
        "a target too many"
    );
    let mut c = cfg(2, 1);
    c.bins = Some(Box::new(crate::BinCfg {
        n_bins: 2,
        edges: Some(vec![vec![0.0], vec![0.0]]),
        rule: crate::BinRule::Quantile,
        warm_rows: 2,
        budget_mib: None,
    }));
    let given = Marginal::new(c).unwrap();
    assert!(!refused(&given, &|_| {}));
    let three = crate::MarginalBins::new(3, 1, vec![vec![0.0]; 3]).unwrap();
    assert!(
        refused(&given, &|b| b.hist = Some(three.clone())),
        "a histogram of three features"
    );
}

/// `beta` is NaN where the feature has no variance, the module doc
/// says: here a spread of 1e-170, whose square underflows to zero while
/// its covariance with a target of unit spread does not. No slope is
/// read from a covariance over a variance of zero.
#[test]
fn a_feature_without_a_variance_has_no_slope() {
    let mut c = cfg(1, 1);
    c.decay = Decay::Halflife(f64::INFINITY);
    let mut m = Marginal::new(c).unwrap();
    for i in 0..6 {
        let x = if i % 2 == 0 { 0.0 } else { 1e-170 };
        OnlineModel::step(&mut m, &[x], &[Some(x * 1e170)], step_clock(i), 1.0);
    }
    let q = m.pair(0, 0);
    assert_eq!(q.var_x, 0.0);
    assert!(q.cov != 0.0 && q.cov.is_finite(), "{}", q.cov);
    assert!(q.beta.is_nan() && q.corr.is_nan(), "{} {}", q.beta, q.corr);
}

/// `t`, `t_serial` and `split_gain_t` read `n − 2` and are NaN at a count
/// of two or less. At Kish's count of exactly two -- weights 1, 1 and 4
/// without decay, `(Σw)²/Σw² = 36/18` -- with a lag deeper than the
/// stream, so every lag correlation is 0, the truncated factor 1 and
/// `n_serial` the same two: each is NaN, where `n − 2 = 0` would have
/// made it 0 over a correlation and a gain that are both short of one.
#[test]
fn a_count_of_two_has_no_statistics() {
    let mut c = cfg(1, 1);
    c.decay = Decay::Halflife(f64::INFINITY);
    c.lags = vec![5];
    c.serial_rule = Some(SerialRule::Truncated);
    c.bins = Some(given_edges(vec![0.5]));
    let mut m = Marginal::new(c).unwrap();
    for (i, (x, y, w)) in [(0.0, 1.0, 1.0), (0.0, 3.0, 1.0), (1.0, 4.0, 4.0)]
        .into_iter()
        .enumerate()
    {
        OnlineModel::step(&mut m, &[x], &[Some(y)], step_clock(i), w);
    }
    let q = m.pair(0, 0);
    assert_eq!((q.n_kish, q.n_serial), (2.0, 2.0));
    assert!(q.corr.is_finite() && q.corr.abs() < 1.0, "{}", q.corr);
    assert!(q.split_gain > 0.0 && q.split_gain < 1.0, "{}", q.split_gain);
    assert!(q.t.is_nan(), "{}", q.t);
    assert!(q.t_serial.is_nan(), "{}", q.t_serial);
    assert!(q.split_gain_t.is_nan(), "{}", q.split_gain_t);
}

/// For a feature that takes two values the best cut separates them, and
/// the stump's gain -- the share of the target's variance the two
/// groups' means explain -- is the regression's `R²`, `corr²`. So
/// `split_gain_t`, the `t` a correlation would need to match the gain,
/// is `|t|` itself: the histogram and the moments, two computations of
/// one number.
#[test]
fn a_two_valued_features_split_statistic_is_its_t() {
    let mut c = cfg(1, 1);
    c.decay = Decay::Halflife(30.0);
    c.bins = Some(given_edges(vec![0.5]));
    let mut m = Marginal::new(c).unwrap();
    let mut s = 8u64;
    for i in 0..200 {
        let x = if lcg(&mut s) > 0.0 { 1.0 } else { 0.0 };
        let y = 0.7 * x + lcg(&mut s);
        let w = 0.5 + lcg(&mut s).abs();
        OnlineModel::step(&mut m, &[x], &[Some(y)], step_clock(i), w);
    }
    let q = m.pair(0, 0);
    assert!(q.t.abs() > 2.0 && q.n_kish > 20.0, "{} {}", q.t, q.n_kish);
    assert!(
        (q.split_gain - q.corr * q.corr).abs() < 1e-12,
        "{}",
        q.split_gain
    );
    let t = q.t.abs();
    assert!(
        (q.split_gain_t - t).abs() < 1e-9 * t,
        "{} against {t}",
        q.split_gain_t
    );
}

/// `t_serial` is `t` at the serial count: `corr·sqrt((n_serial − 2)/(1 −
/// corr²))`, so `t_serial²/(n_serial − 2) = t²/(n_kish − 2)`, with
/// `n_serial` Kish's count over the factor the pair's own lag
/// correlations give, `1 + 2·Σ_l ρ_x(l)·ρ_y(l)` under `"truncated"`.
#[test]
fn t_serial_is_t_at_the_serial_count() {
    let mut c = cfg(1, 1);
    c.decay = Decay::Halflife(200.0);
    c.lags = vec![1, 2, 3];
    c.serial_rule = Some(SerialRule::Truncated);
    let mut m = Marginal::new(c).unwrap();
    let mut s = 21u64;
    let (mut x, mut y) = (0.0, 0.0);
    for i in 0..600 {
        x = 0.8 * x + normal(&mut s);
        y = 0.6 * y + 0.3 * x + normal(&mut s);
        OnlineModel::step(&mut m, &[x], &[Some(y)], step_clock(i), 1.0);
    }
    let q = m.pair(0, 0);
    let products: f64 = (0..3).map(|l| q.lagcorr_xx[l] * q.lagcorr_yy[l]).sum();
    let factor = 1.0 + 2.0 * products;
    assert!(factor > 1.2, "serially dependent: {factor}");
    assert!((q.n_serial - q.n_kish / factor).abs() < 1e-9 * q.n_serial);
    let want = q.corr * ((q.n_serial - 2.0) / (1.0 - q.corr * q.corr)).sqrt();
    let t_serial = q.t_serial;
    assert!(
        (t_serial - want).abs() < 1e-12 * want.abs(),
        "{t_serial} against {want}"
    );
    let ratio = t_serial / q.t;
    let counts = (q.n_serial - 2.0) / (q.n_kish - 2.0);
    assert!((ratio * ratio - counts).abs() < 1e-12, "{ratio} {counts}");
}

/// A serial factor that is not a finite number is no correction: a
/// state whose lag moments were edited so far from the variances that
/// `ρ_x·ρ_y` overflows reports `n_serial` NaN, not `n_kish/∞ = 0`.
#[test]
fn an_infinite_serial_factor_is_no_correction() {
    let mut c = cfg(1, 1);
    c.lags = vec![1];
    c.serial_rule = Some(SerialRule::Truncated);
    let mut m = Marginal::new(c).unwrap();
    let mut s = 4u64;
    for i in 0..30 {
        let x = lcg(&mut s);
        OnlineModel::step(&mut m, &[x], &[Some(x + lcg(&mut s))], step_clock(i), 1.0);
    }
    assert!(m.pair(0, 0).n_serial.is_finite());
    let mut v = serde_json::to_value(&m).unwrap();
    v["lag"]["cxx"][0][0] = serde_json::json!(1e300);
    v["lag"]["cyy"][0][0] = serde_json::json!(1e300);
    let edited: Marginal = serde_json::from_value(v).unwrap();
    let q = edited.pair(0, 0);
    assert_eq!(q.lagcorr_xx[0] * q.lagcorr_yy[0], f64::INFINITY);
    assert!(q.n_serial.is_nan() && q.t_serial.is_nan(), "{}", q.n_serial);
}

/// `"auto"`'s cost model: a lag costs `LAG_UNITS` of a pair and a cross
/// lag two more, its two terms, so three lags with no cross terms are
/// the work of one lag with its own. At `p·T = 1,000` and 256 rows a
/// pair of `0.6·(1 + 0.4·3)` ns makes a flush of 337,920 ns: three shards
/// of 100 µs.
#[test]
fn a_cross_lag_costs_two_lags() {
    let shape = |lags: Vec<usize>, cross: Option<Vec<usize>>| {
        let mut c = cfg(100, 10);
        c.lags = lags;
        c.cross_lags = cross;
        c
    };
    let three = shape(vec![1, 2, 3], Some(vec![]));
    let one = shape(vec![1], None);
    assert_eq!(batch_rows(100), 256);
    assert_eq!(three.auto_shards(8), 3);
    assert_eq!(one.auto_shards(8), 3);
}

/// Held rows make a model equal to nothing, a copy of itself included,
/// until they are flushed (`Deferred`'s `eq`). A row of no weight moves
/// no pair, so once flushed the model is the plain step's, and before
/// it only the held row told the two apart.
#[test]
fn a_model_with_rows_held_equals_nothing() {
    let shards = Shards {
        count: 2,
        run: &run_in_order,
    };
    let mut plain = Marginal::new(cfg(4, 1)).unwrap();
    let mut s = 6u64;
    for i in 0..10 {
        let x: Vec<f64> = (0..4).map(|_| lcg(&mut s)).collect();
        OnlineModel::step(&mut plain, &x, &[Some(x[0] - x[1])], step_clock(i), 1.0);
    }
    let mut held = plain.clone();
    let row = [0.5, -0.5, 2.0, 1.0];
    held.step_sharded(&row, &[Some(1.0)], 0.0, 0.0, &shards);
    OnlineModel::step(&mut plain, &row, &[Some(1.0)], 0.0, 0.0);
    assert_eq!(held.held_rows(), 1);
    assert_ne!(held, held.clone());
    assert_ne!(held, plain);
    held.flush(&shards);
    assert_eq!(held, plain);
}

/// A full batch is flushed keeping its buffers for the next one, and the
/// caller's flush lets them go ([`Marginal::flush`], `Deferred::clear`):
/// a group that stops holds no batch's worth of memory.
#[test]
fn a_batch_keeps_its_buffers_and_the_flush_lets_them_go() {
    let shards = Shards {
        count: 2,
        run: &run_in_order,
    };
    let p = 4;
    let batch = batch_rows(p);
    let mut m = Marginal::new(cfg(p, 1)).unwrap();
    let mut s = 12u64;
    for i in 0..batch + 3 {
        let x: Vec<f64> = (0..p).map(|_| lcg(&mut s)).collect();
        m.step_sharded(&x, &[Some(x[0])], step_clock(i), 1.0, &shards);
    }
    assert_eq!(m.held_rows(), 3, "the full batch was flushed");
    assert!(
        m.defer.xs.capacity() >= batch * p,
        "{}",
        m.defer.xs.capacity()
    );
    m.flush(&shards);
    assert_eq!(m.defer.xs.capacity(), 0);
}

/// The window's shadow foresees its ring: under a refusing budget it
/// crosses on the first snapshot, the shadow learning the same clock
/// steps says it is over exactly when the model does, and nothing is
/// over before. Without a window there is no ring, and nothing over.
#[test]
fn the_windows_shadow_foresees_its_overrun() {
    let mut c = cfg(2, 1);
    c.window = Some(1e9);
    let mut m = Marginal::new(c).unwrap();
    m.set_window_budget(Some(crate::WindowBudget::Refuse(1e-9)));
    assert_eq!(m.window_over_budget(), None);
    let mut shadow = m.window_shadow().expect("a window has a ring");
    assert_eq!(shadow.over_budget(), None);
    for i in 0..4 {
        shadow.learn(step_clock(i), None);
        let v = i as f64;
        OnlineModel::step(&mut m, &[v, 1.0], &[Some(v)], step_clock(i), 1.0);
        assert_eq!(shadow.over_budget(), m.window_over_budget(), "row {i}");
    }
    let (bytes, every) = m.window_over_budget().expect("the budget was crossed");
    assert!(
        bytes > 0 && every == crate::Cadence::EVERY_ROW,
        "{bytes} {every:?}"
    );
    let plain = Marginal::new(cfg(2, 1)).unwrap();
    assert!(plain.window_shadow().is_none());
    assert_eq!(plain.window_over_budget(), None);
}

/// One corruption per condition `restore` checks, each refused alone:
/// `min_weight`'s length, lags the lag moments were not kept at, bins in
/// the cfg and none in the state, a window snapshot short a target's
/// weight, a feature's mean or a covariance, and a window the cfg does
/// not have. A windowed state at `p = 3`, `T = 2`, where a pair
/// vector's `p·T` is not `p + T`, restores as it is.
#[test]
fn each_part_of_a_state_is_checked_alone() {
    use crate::ModelState;
    let run = |c: MarginalCfg| {
        let mut m = Marginal::new(c).unwrap();
        let mut s = 9u64;
        for i in 0..12 {
            let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
            let y = [Some(x[0] + x[2]), (i % 2 == 0).then_some(x[1])];
            OnlineModel::step(&mut m, &x, &y, step_clock(i), 1.0);
        }
        m
    };
    let mut c = cfg(3, 2);
    c.window = Some(5.0);
    let windowed = run(c);
    let mut c = cfg(3, 2);
    c.lags = vec![1, 2];
    let lagged = run(c);
    let snapshot = |m: &mut Marginal| {
        let win = m.win.as_mut().unwrap();
        win.snaps.iter_mut().next().expect("a snapshot").clone()
    };
    assert!(snapshot(&mut windowed.clone()).sxy.len() == 6);
    let restored = |m: &Marginal, f: &dyn Fn(&mut Marginal)| {
        let mut s = m.state();
        let ModelState::Marginal(inner) = &mut s.model else {
            unreachable!()
        };
        f(inner);
        Marginal::restore(&s)
    };
    assert_eq!(
        state_bytes(&restored(&windowed, &|_| {}).unwrap()),
        state_bytes(&windowed)
    );
    assert!(restored(&lagged, &|_| {}).is_ok());
    fn first(m: &mut Marginal) -> &mut MarginalMoments {
        m.win.as_mut().unwrap().snaps.iter_mut().next().unwrap()
    }
    // A threshold short a target is the cfg's own refusal, by name, since
    // every `restore` runs the cfg's check (review 2026-10-06, CF4).
    match restored(&windowed, &|m| {
        m.cfg.min_weight.pop();
    }) {
        Err(StateError::Invalid(e)) => assert!(e.contains("min_weight has 1 entries"), "{e}"),
        other => panic!("min_weight short a target: {other:?}"),
    }
    type Case<'a> = (&'a str, &'a Marginal, &'a dyn Fn(&mut Marginal));
    let cases: [Case; 6] = [
        ("lags the moments were not kept at", &lagged, &|m| {
            m.cfg.lags = vec![1, 3]
        }),
        // On the unwindowed model: bins beside a window is a cfg `new`
        // refuses, which `restore` refuses as that now.
        ("bins in the cfg alone", &lagged, &|m| {
            m.cfg.bins = Some(bins_cfg(3, 50))
        }),
        ("a snapshot short a weight", &windowed, &|m| {
            first(m).wt.pop();
        }),
        ("a snapshot short a mean", &windowed, &|m| {
            first(m).mx.pop();
        }),
        ("a snapshot short a covariance", &windowed, &|m| {
            first(m).sxy.pop();
        }),
        ("a window the cfg does not have", &windowed, &|m| {
            m.cfg.window = None;
            m.cfg.window_every = None;
        }),
    ];
    for (what, m, corrupt) in cases {
        match restored(m, corrupt) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{what}: {e}"),
            other => panic!("{what}: {other:?}"),
        }
    }
}

/// [`a_sharded_step_is_the_unsplit_step_to_the_bit`] on rows whose
/// features and targets are all finite, as the bank's are (it skips a
/// row with a feature that is not). The stream there builds its targets
/// from features that are NaN now and then, so from its first rows every
/// target, and with it every pair, is NaN -- and a NaN's bytes are a
/// NaN's, so most of what it compares is NaN against NaN. Here every
/// pair is a number to the end, which the test checks first. Beside
/// [`shard_cfgs`], shared feature moments with every cross lag, and a
/// window shorter than its snapshot cadence, whose snapshots come from
/// the clock rather than the count; and now and then a plain step
/// between sharded ones, which learns the held rows first.
#[test]
fn a_sharded_step_is_the_unsplit_step_on_finite_rows() {
    let p = 7;
    let rows = shard_rows(500, p, false);
    let mut cfgs = shard_cfgs(p);
    let base = cfgs[0].1.clone();
    cfgs.push((
        "shared with every cross lag",
        MarginalCfg {
            feature_moments: FeatureMomentLayout::Shared,
            lags: vec![1, 3, 8],
            serial_rule: Some(SerialRule::Geometric),
            ..base.clone()
        },
    ));
    cfgs.push((
        "a window shorter than its cadence",
        MarginalCfg {
            window: Some(2.5),
            max_rows_between_snapshots: Some(5),
            ..base.clone()
        },
    ));
    cfgs.push((
        "a window shorter than its clock spacing",
        MarginalCfg {
            window: Some(2.5),
            window_every: Some(5.0),
            ..base
        },
    ));
    let runners: [(&str, &ShardRunner<'_>); 2] =
        [("in order", &run_in_order), ("reversed", &reversed)];
    for (name, c) in cfgs {
        let mut plain = Marginal::new(c.clone()).unwrap();
        for r in &rows {
            if r.clear {
                OnlineModel::clear_lags(&mut plain);
            }
            OnlineModel::step(&mut plain, &r.x, &r.y, r.d, r.w);
        }
        for t in 0..3 {
            for j in 0..p {
                let q = plain.pair(t, j);
                assert!(
                    q.mean_x.is_finite() && q.cov.is_finite(),
                    "{name}: pair ({t}, {j}) is {q:?}"
                );
            }
        }
        let want = state_bytes(&plain);
        for count in [2, 3, 50] {
            for (how, run) in runners {
                let shards = Shards { count, run };
                let mut m = Marginal::new(c.clone()).unwrap();
                let mut plain_with_held = 0;
                for (i, r) in rows.iter().enumerate() {
                    if r.clear {
                        OnlineModel::clear_lags(&mut m);
                    }
                    if i % 61 == 30 {
                        plain_with_held += usize::from(m.held_rows() > 0);
                        OnlineModel::step(&mut m, &r.x, &r.y, r.d, r.w);
                    } else {
                        m.step_sharded(&r.x, &r.y, r.d, r.w, &shards);
                    }
                }
                m.flush(&shards);
                assert!(
                    state_bytes(&m) == want,
                    "{name}, {count} shards {how}: the state differs"
                );
                if !name.starts_with("window") && !name.contains("window") {
                    assert!(plain_with_held > 3, "{name}: {plain_with_held}");
                }
            }
        }
    }
}

/// A row with no finite feature writes no cell, so an empty histogram
/// stays empty and its next row's ageing is skipped -- on a sharded step
/// as on the plain one, where a sharded row marked the histogram
/// written ahead of its cells.
#[test]
fn a_row_with_no_finite_feature_leaves_the_histogram_empty_when_sharded() {
    let mut c = cfg(2, 1);
    c.bins = Some(Box::new(crate::BinCfg {
        n_bins: 2,
        edges: Some(vec![vec![0.0], vec![0.0]]),
        rule: crate::BinRule::Quantile,
        warm_rows: 2,
        budget_mib: None,
    }));
    let shards = Shards {
        count: 2,
        run: &run_in_order,
    };
    let mut plain = Marginal::new(c.clone()).unwrap();
    let mut sharded = Marginal::new(c).unwrap();
    let rows = [
        ([f64::NAN, f64::NAN], 1.0, 0.0),
        ([0.5, -0.5], 2.0, 1.0),
        ([-1.0, 1.0], 0.5, 1.0),
        ([2.0, 0.1], 1.5, 1.0),
    ];
    for (x, y, d) in rows {
        OnlineModel::step(&mut plain, &x, &[Some(y)], d, 1.0);
        sharded.step_sharded(&x, &[Some(y)], d, 1.0, &shards);
    }
    assert!(sharded.held_rows() > 0);
    sharded.flush(&shards);
    assert!(state_bytes(&plain) == state_bytes(&sharded));
}

/// A row with a value that is not usable takes no slot in the lag ring and
/// no row of the bins' warm-up hold (`OnlineModel`, task 183), as a row of
/// weight 0 takes neither: the rows either side of it are adjacent in the
/// ring, as they are when the plumbing skips the row, and the hold only
/// ages over it. The ring and the hold took no row with a feature that was
/// not finite before either, but the pairs learned such a row, its NaN
/// never leaving them, and a feature past the input bound was taken by all
/// three. Through `step` and `step_sharded` alike, which holds the refused
/// row in its batch as `step` learns it.
#[test]
fn a_refused_row_takes_no_ring_slot_and_no_warm_up_row() {
    use crate::OnlineModel;
    let mut c = cfg(2, 1);
    c.lags = vec![1, 2];
    c.bins = Some(bins_cfg(4, 30));
    let ring = |m: &Marginal| -> Vec<(Vec<f64>, Vec<Option<f64>>)> {
        let lag = m.lag.as_ref().unwrap();
        (0..lag.depth())
            .map(|i| {
                let (x, y) = lag.ring_row(i);
                (x.to_vec(), y.to_vec())
            })
            .collect()
    };
    let held = |m: &Marginal| m.bins.as_ref().unwrap().held.len();
    let shards = Shards {
        count: 3,
        run: &run_in_order,
    };
    for (bad, w) in [
        ([f64::NAN, 0.5], 1.0),
        ([0.5, 2.0 * crate::INPUT_BOUND], 1.0),
        ([0.5, 0.25], f64::NAN),
    ] {
        for sharded in [false, true] {
            let case = format!("{bad:?} at weight {w}, sharded {sharded}");
            let step = |m: &mut Marginal, x: &[f64], y: &[Option<f64>], d: f64, w: f64| {
                if sharded {
                    m.step_sharded(x, y, d, w, &shards)
                } else {
                    OnlineModel::step(m, x, y, d, w)
                }
            };
            let mut with = Marginal::new(c.clone()).unwrap();
            let mut without = Marginal::new(c.clone()).unwrap();
            let mut s = 31u64;
            for i in 0..12 {
                let x = [lcg(&mut s), lcg(&mut s)];
                let y = [Some(x[0] - 0.5 * x[1])];
                let d = if i == 0 { 0.0 } else { 1.0 };
                step(&mut with, &x, &y, d, 1.0);
                step(&mut without, &x, &y, d, 1.0);
            }
            with.flush(&shards);
            let (ring_before, held_before) = (ring(&with), held(&with));
            assert!(
                ring_before.len() == 2 && held_before == 12,
                "{case}: the fixture"
            );
            step(&mut with, &bad, &[Some(0.3)], 1.0, w);
            with.flush(&shards);
            assert_eq!(ring(&with), ring_before, "{case}: the ring took the row");
            assert_eq!(held(&with), held_before, "{case}: the hold took the row");
            let x = [0.3, -0.7];
            step(&mut with, &x, &[Some(0.65)], 1.0, 1.0);
            step(&mut without, &x, &[Some(0.65)], 2.0, 1.0);
            with.flush(&shards);
            without.flush(&shards);
            assert_eq!(ring(&with), ring(&without), "{case}: the rows either side");
            assert_eq!(held(&with), held(&without), "{case}");
        }
    }
}
