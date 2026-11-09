use super::*;

/// A zero-weight head row ages the prior's scale as it advances the
/// clock: `update` at `W == 0` with `w == 0` is the decay, as `rls` ages
/// its factor on every row; refusing it left the scale at 1 (task 159, R1).
#[test]
fn a_zero_weight_head_row_ages_the_prior() {
    let mut c = EwCov::new(1);
    c.update(&[1.0], 0.5, 0.0);
    assert_eq!((c.w_sum, c.prior_scale()), (0.0, 0.5));
    c.update(&[1.0], 0.5, 0.0);
    assert_eq!((c.w_sum, c.prior_scale()), (0.0, 0.25));
    c.update(&[1.0], 0.5, 1.0);
    assert_eq!((c.w_sum, c.prior_scale()), (1.0, 0.125));
}

/// A state whose accumulator is not the cfg's width is refused, where
/// it loaded and panicked on the first `step` (review 2026-09-18, B3).
#[test]
fn a_state_of_the_wrong_shape_is_refused() {
    use crate::{ModelState, OnlineModel, StateError};
    let m = EwCovModel::new(model_cfg(2, vec![EwCovStat::Mean])).unwrap();
    let mut s = m.state();
    let ModelState::EwCovModel(inner) = &mut s.model else {
        unreachable!()
    };
    inner.cov = EwCov::new(3);
    match EwCovModel::restore(&s) {
        Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
        other => panic!("{other:?}"),
    }
}

/// Direct O(n^2) recomputation with explicit decayed weights.
fn direct(xs: &[Vec<f64>], lams: &[f64], ws: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
    let n = xs.len();
    let k = xs[0].len();
    let mut wts = vec![0.0; n];
    for i in 0..n {
        let mut wi = ws[i];
        for lam in &lams[i + 1..] {
            wi *= lam;
        }
        wts[i] = wi;
    }
    let wsum: f64 = wts.iter().sum();
    let mut m = vec![0.0; k];
    let mut s = vec![0.0; k * k];
    for (x, &wi) in xs.iter().zip(&wts) {
        for i in 0..k {
            m[i] += wi * x[i] / wsum;
            for j in 0..k {
                s[i * k + j] += wi * x[i] * x[j] / wsum;
            }
        }
    }
    (wsum, m, s)
}

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn model_cfg(k: usize, stats: Vec<EwCovStat>) -> EwCovCfg {
    EwCovCfg {
        bias: true,
        n_features: k,
        decay: crate::Decay::Halflife(f64::INFINITY),
        stats,
        min_weight: 2.0,
        precision_prior: None,
        mahal_quantiles: Vec::new(),
        pca: 0,
        pca_every: 0.0,
        max_rows_between_pca: u32::MAX,
        lags: Vec::new(),
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

/// Whether a row `age` clock units old is inside a `window` with edge
/// `closed`, by the definition (docs/PLAN.md task 196, N17): less than
/// the window under `Right`, at most it under `Both`.
fn inside(age: f64, window: f64, closed: crate::WindowClosed) -> bool {
    match closed {
        crate::WindowClosed::Right => age < window,
        crate::WindowClosed::Both => age <= window,
    }
}

/// A windowed weighted mean and covariance, written from the definition:
/// every row inside the window, weighted by `lam^age`, and nothing else.
/// A row exactly `window` old is inside under `closed = "both"` and has
/// left under `"right"`, the default ([`inside`]).
/// `t` are absolute clocks; the report at row `n` is referenced at the
/// clock of row `n-1`, as every statistic in this library is.
fn direct_window(
    xs: &[Vec<f64>],
    t: &[f64],
    half_life: f64,
    window: f64,
    upto: usize,
    closed: crate::WindowClosed,
) -> (f64, Vec<f64>, Vec<f64>) {
    let k = xs[0].len();
    let now = t[upto - 1];
    let mut w_sum = 0.0;
    let mut m = vec![0.0; k];
    let mut raw = vec![0.0; k * k];
    for i in 0..upto {
        if !inside(now - t[i], window, closed) {
            continue;
        }
        let w = 0.5_f64.powf((now - t[i]) / half_life);
        w_sum += w;
        for a in 0..k {
            m[a] += w * xs[i][a];
            for b in 0..k {
                raw[a * k + b] += w * xs[i][a] * xs[i][b];
            }
        }
    }
    for v in m.iter_mut() {
        *v /= w_sum;
    }
    let mut cen = vec![0.0; k * k];
    for a in 0..k {
        for b in 0..k {
            cen[a * k + b] = raw[a * k + b] / w_sum - m[a] * m[b];
        }
    }
    (w_sum, m, cen)
}

/// PLAN §13.4 (1): the truncated view against a direct windowed sum,
/// under either edge (docs/PLAN.md task 196, N17).
#[test]
fn a_window_reports_exactly_the_rows_inside_it() {
    let half_life = 40.0;
    for (closed, &window) in [crate::WindowClosed::Right, crate::WindowClosed::Both]
        .into_iter()
        .flat_map(|c| [15.0, 60.0, 150.0].iter().map(move |w| (c, w)))
    {
        let mut cfg = model_cfg(2, vec![EwCovStat::Mean, EwCovStat::Var, EwCovStat::Cov]);
        cfg.decay = crate::Decay::Halflife(half_life);
        cfg.min_weight = 0.0;
        cfg.window = Some(window);
        let mut m = EwCovModel::new(cfg).unwrap();
        crate::OnlineModel::set_window_closed(&mut m, closed);

        let (mut xs, mut t, mut clock) = (Vec::new(), Vec::new(), 0.0);
        let mut on_the_boundary = 0;
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f64) / (u32::MAX as f64)
        };
        for i in 0..120 {
            // Quarter units, exact in a double, so some rows land
            // exactly one window old (the boundary is inclusive).
            let d = if i == 0 {
                0.0
            } else {
                0.25 * (4.0 + (rnd() * 24.0).floor())
            };
            clock += d;
            let x = vec![rnd() * 4.0 - 2.0, rnd() * 10.0 + 100.0];
            if i > 0 {
                // The report is referenced at the previous row's clock.
                let got = crate::OnlineModel::predict(&m, &x, d);
                let (w, mean, cen) = direct_window(&xs, &t, half_life, window, i, closed);
                on_the_boundary += t.iter().filter(|&&tj| t[i - 1] - tj == window).count();
                assert!(
                    (got.n_eff - w).abs() < 1e-9 * w.max(1.0),
                    "window {window} row {i}: n_eff {} vs {w}",
                    got.n_eff
                );
                for a in 0..2 {
                    assert!(
                        (got.pred[a] - mean[a]).abs() < 1e-9,
                        "window {window} row {i} mean[{a}]: {} vs {}",
                        got.pred[a],
                        mean[a]
                    );
                    let var = cen[a * 2 + a];
                    assert!(
                        (got.pred[2 + a] - var).abs() < 1e-8 * var.abs().max(1.0),
                        "window {window} row {i} var[{a}]: {} vs {var}",
                        got.pred[2 + a]
                    );
                }
                assert!(
                    (got.pred[4] - cen[1]).abs() < 1e-8 * cen[1].abs().max(1.0),
                    "window {window} row {i} cov: {} vs {}",
                    got.pred[4],
                    cen[1]
                );
            }
            crate::OnlineModel::step(&mut m, &x, &[], d, 1.0);
            xs.push(x);
            t.push(clock);
        }
        // The boundary was exercised, not only the rows either side of it.
        assert!(
            on_the_boundary >= 5,
            "window {window}: {on_the_boundary} reports had a row exactly one window old"
        );
    }
}

/// Review 2026-09-12, C17: the same window at a level of `1e8`. The old
/// truncation went back through `E[x x'] = C + m m'`, subtracted, and
/// re-centred, which at this level left the windowed variance with a
/// resolution of about 2 -- nothing, for unit-scale data. The oracle here
/// is two-pass (centre on the window's own mean, then square), so it is
/// right at any offset; `direct_window` above is not, which is why it runs
/// at `100` and this one needs its own. A row exactly `window` old is
/// inside under `closed = "both"` and has left under `"right"`, the
/// default (docs/PLAN.md task 196): both are run.
#[test]
fn a_window_keeps_its_precision_at_a_large_offset() {
    let half_life = 40.0;
    for (closed, &window) in [crate::WindowClosed::Right, crate::WindowClosed::Both]
        .into_iter()
        .flat_map(|c| [15.0, 60.0, 150.0].iter().map(move |w| (c, w)))
    {
        let mut cfg = model_cfg(2, vec![EwCovStat::Mean, EwCovStat::Var, EwCovStat::Cov]);
        cfg.decay = crate::Decay::Halflife(half_life);
        cfg.min_weight = 0.0;
        cfg.window = Some(window);
        let mut m = EwCovModel::new(cfg).unwrap();
        crate::OnlineModel::set_window_closed(&mut m, closed);

        let (mut xs, mut t, mut clock) = (Vec::<Vec<f64>>::new(), Vec::new(), 0.0);
        let mut on_the_boundary = 0;
        let mut seed = 777u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f64) / (u32::MAX as f64)
        };
        for i in 0..160 {
            // Quarter units, exact in a double, so some rows land
            // exactly one window old (the boundary is inclusive).
            let d = if i == 0 {
                0.0
            } else {
                0.25 * (4.0 + (rnd() * 24.0).floor())
            };
            clock += d;
            let u = rnd() * 4.0 - 2.0;
            let x = vec![1e8 + u, 1e8 + 0.5 * u + rnd()];
            if i > 0 {
                let got = crate::OnlineModel::predict(&m, &x, d);
                // S15: the accessor is the windowed weight the report
                // carries, not the live one.
                assert!(
                    (m.n_eff() - got.n_eff).abs() < 1e-12 * got.n_eff.max(1.0),
                    "window {window} row {i}: accessor {} vs emitted {}",
                    m.n_eff(),
                    got.n_eff
                );
                let now = t[i - 1];
                let keep: Vec<usize> = (0..i)
                    .filter(|&j| inside(now - t[j], window, closed))
                    .collect();
                on_the_boundary += (0..i).filter(|&j| now - t[j] == window).count();
                let w: Vec<f64> = keep
                    .iter()
                    .map(|&j| 0.5_f64.powf((now - t[j]) / half_life))
                    .collect();
                let wsum: f64 = w.iter().sum();
                let mean: Vec<f64> = (0..2)
                    .map(|a| {
                        keep.iter()
                            .zip(&w)
                            .map(|(&j, wj)| wj * xs[j][a])
                            .sum::<f64>()
                            / wsum
                    })
                    .collect();
                let cen = |a: usize, b: usize| {
                    keep.iter()
                        .zip(&w)
                        .map(|(&j, wj)| wj * (xs[j][a] - mean[a]) * (xs[j][b] - mean[b]))
                        .sum::<f64>()
                        / wsum
                };
                // Unit-scale spreads, so an absolute 1e-6 separates the
                // fix (errors near 1e-9, the live accumulator's own at this
                // level) from the defect (errors of order 1).
                let close = |got: f64, want: f64| (got - want).abs() < 1e-6 * want.abs().max(1.0);
                assert!(
                    (got.n_eff - wsum).abs() < 1e-9 * wsum,
                    "window {window} row {i}: n_eff {} vs {wsum}",
                    got.n_eff
                );
                for (a, &want) in mean.iter().enumerate() {
                    assert!(
                        (got.pred[a] - want).abs() < 1e-12 * want.abs(),
                        "window {window} row {i} mean[{a}]: {} vs {want}",
                        got.pred[a]
                    );
                    assert!(
                        close(got.pred[2 + a], cen(a, a)),
                        "window {window} row {i} var[{a}]: {} vs {}",
                        got.pred[2 + a],
                        cen(a, a)
                    );
                }
                assert!(
                    close(got.pred[4], cen(0, 1)),
                    "window {window} row {i} cov: {} vs {}",
                    got.pred[4],
                    cen(0, 1)
                );
            }
            crate::OnlineModel::step(&mut m, &x, &[], d, 1.0);
            xs.push(x);
            t.push(clock);
        }
        assert!(
            on_the_boundary >= 5,
            "window {window}: {on_the_boundary} reports had a row exactly one window old"
        );
    }
}

/// PLAN §13.4 (2): the guarantee itself. Rows older than the window are
/// replaced with a number that would dominate any average they still
/// touched.
#[test]
fn nothing_older_than_the_window_can_move_the_answer() {
    let run = |poison: bool| {
        let mut cfg = model_cfg(1, vec![EwCovStat::Mean, EwCovStat::Var]);
        cfg.decay = crate::Decay::Halflife(20.0);
        cfg.min_weight = 0.0;
        cfg.window = Some(60.0);
        let mut m = EwCovModel::new(cfg).unwrap();
        let mut last = crate::Step {
            pred: vec![],
            n_eff: 0.0,
            extra: None,
        };
        for i in 0..200 {
            // 200 rows one clock unit apart. The last report is read at
            // the clock of row 198 and its 60-unit window covers rows 139
            // and after, so rows before 130 are outside it with room to
            // spare. (Poisoning from 140 puts 1e6 *inside* the window,
            // which is what the first version of this test did.)
            let inside = i >= 130;
            let x = if poison && !inside {
                vec![1e6]
            } else {
                vec![(i as f64 * 0.37).sin()]
            };
            last = crate::OnlineModel::step(&mut m, &x, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        last
    };
    let clean = run(false);
    let dirty = run(true);
    assert!(
        (clean.pred[0] - dirty.pred[0]).abs() < 1e-8,
        "1e6 outside the window moved the mean: {} vs {}",
        clean.pred[0],
        dirty.pred[0]
    );
    assert!(
        (clean.n_eff - dirty.n_eff).abs() < 1e-9,
        "n_eff moved: {} vs {}",
        clean.n_eff,
        dirty.n_eff
    );
}

/// PLAN §13.4 (5): a window longer than the stream is the ordinary
/// accumulator, slot for slot, so the feature cannot change what an
/// existing spec reports.
#[test]
fn a_window_no_stream_reaches_is_the_untruncated_model() {
    let mk = |window: Option<f64>| {
        let mut cfg = model_cfg(2, vec![EwCovStat::Mean, EwCovStat::Var, EwCovStat::Corr]);
        cfg.decay = crate::Decay::Halflife(25.0);
        cfg.min_weight = 0.0;
        cfg.window = window;
        EwCovModel::new(cfg).unwrap()
    };
    let (mut plain, mut windowed) = (mk(None), mk(Some(1e9)));
    for i in 0..80 {
        let x = vec![(i as f64 * 0.7).sin(), (i as f64 * 0.31).cos() * 3.0];
        let d = if i == 0 { 0.0 } else { 1.0 };
        let a = crate::OnlineModel::step(&mut plain, &x, &[], d, 1.0);
        let b = crate::OnlineModel::step(&mut windowed, &x, &[], d, 1.0);
        assert!(
            same_bits(&a.pred, &b.pred),
            "row {i}: {:?} vs {:?}",
            a.pred,
            b.pred
        );
        assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits(), "row {i} n_eff");
    }
}

/// `window_every` trades memory for a boundary that can only move
/// *inward*: a coarser cadence never keeps a row the window excludes.
/// Ten clock units here, a row each.
#[test]
fn a_coarse_cadence_discards_more_and_never_less() {
    let mk = |every: f64| {
        let mut cfg = model_cfg(1, vec![EwCovStat::Mean]);
        cfg.decay = crate::Decay::Halflife(30.0);
        cfg.min_weight = 0.0;
        cfg.window = Some(50.0);
        cfg.window_every = Some(every);
        EwCovModel::new(cfg).unwrap()
    };
    let (mut fine, mut coarse) = (mk(0.0), mk(10.0));
    for i in 0..300 {
        // A step function: old rows are 0, recent rows are 1. A window
        // that reaches further back reports a *smaller* mean.
        let x = vec![if i < 200 { 0.0 } else { 1.0 }];
        let d = if i == 0 { 0.0 } else { 1.0 };
        let a = crate::OnlineModel::step(&mut fine, &x, &[], d, 1.0);
        let b = crate::OnlineModel::step(&mut coarse, &x, &[], d, 1.0);
        if i > 210 {
            assert!(
                b.pred[0] >= a.pred[0] - 1e-12,
                "row {i}: the coarse window kept more of the old regime: {} < {}",
                b.pred[0],
                a.pred[0]
            );
            assert!(
                b.n_eff <= a.n_eff + 1e-12,
                "row {i}: coarse n_eff is larger"
            );
        }
    }
}

/// The window's snapshot counts every vector it holds in its footprint
/// (docs/PLAN.md task 130).
#[test]
fn the_window_footprint_counts_every_vector() {
    let snap = |k: usize| {
        let mut cfg = model_cfg(k, vec![EwCovStat::Mean, EwCovStat::Cov]);
        cfg.window = Some(20.0);
        let mut m = EwCovModel::new(cfg).unwrap();
        let mut s = 17u64;
        for i in 0..40 {
            let x: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
            let d = if i == 0 { 0.0 } else { 1.0 };
            crate::OnlineModel::step(&mut m, &x, &[], d, 1.0);
        }
        m.win.as_ref().unwrap().snaps.boundary().unwrap().1.clone()
    };
    crate::window::assert_footprint_counts_every_vector(&snap(2), &snap(5), "ew_cov");
}

/// A state from before the runs' flag (0.10.0) restored under a spec
/// without a window keeps no runs, as a model built under it keeps none;
/// and a state whose runs are off under a spec with a window is refused,
/// since the window would read every held slot as moving (review
/// 2026-09-26, C4).
#[test]
fn restore_sets_the_runs_from_the_window() {
    let build = |window: Option<f64>| {
        let mut cfg = model_cfg(2, vec![EwCovStat::Mean]);
        cfg.window = window;
        let mut m = EwCovModel::new(cfg).unwrap();
        for i in 0..12 {
            let x = [i as f64, 1.0];
            crate::OnlineModel::step(&mut m, &x, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        serde_json::to_value(crate::OnlineModel::state(&m)).unwrap()
    };
    let restore = |v: serde_json::Value| {
        <EwCovModel as crate::OnlineModel>::restore(&serde_json::from_value(v).unwrap())
    };
    let mut v = build(None);
    crate::window::json_edit(&mut v, "runs", &mut |x| {
        *x = serde_json::json!({"x": [], "start": []});
    });
    assert!(
        !restore(v).unwrap().cov().keeps_runs(),
        "without the flag, under no window"
    );
    let mut v = build(Some(5.0));
    crate::window::json_edit(&mut v, "runs", &mut |x| {
        x["off"] = serde_json::json!(true);
    });
    assert!(matches!(restore(v), Err(crate::StateError::Invalid(_))));
}

/// The cadence counts every row the model is stepped with, rows of
/// weight zero included: a stream alternating weight 1 and 0 under
/// `max_rows_between_snapshots = 2` snapshots once a row pair, not once
/// a learned row (review 2026-09-26, C7; the docs said learned rows),
/// and a row of weight zero advances the clock `window_every` is
/// measured on, two units here, a unit a row (docs/PLAN.md task 162).
#[test]
fn a_zero_weight_row_counts_toward_the_snapshot_cadence() {
    for (every, rows) in [(None, Some(2)), (Some(2.0), None)] {
        let mut cfg = model_cfg(1, vec![EwCovStat::Mean]);
        cfg.decay = crate::Decay::Halflife(30.0);
        cfg.window = Some(1e9);
        cfg.window_every = every;
        cfg.max_rows_between_snapshots = rows;
        let mut m = EwCovModel::new(cfg).unwrap();
        for i in 0..40 {
            let w = if i % 2 == 0 { 1.0 } else { 0.0 };
            let d = if i == 0 { 0.0 } else { 1.0 };
            crate::OnlineModel::step(&mut m, &[i as f64], &[], d, w);
        }
        assert_eq!(
            m.win.as_ref().unwrap().snaps.len(),
            20,
            "{every:?}, {rows:?}: one snapshot per two rows"
        );
    }
}

#[test]
fn model_cfg_validation_rejects_each_bad_field() {
    use EwCovStat::*;
    let err = |c: EwCovCfg, want: &str| match c.validate() {
        Err(e) => assert!(e.contains(want), "wanted {want:?}, got {e:?}"),
        Ok(()) => panic!("expected rejection mentioning {want:?}"),
    };

    err(model_cfg(0, vec![Mean]), "needs at least one feature");
    // No statistics is "accumulate only" (E43), not an error.
    model_cfg(2, vec![]).validate().unwrap();
    assert_eq!(model_cfg(2, vec![]).n_outputs(), 0);

    // Per-column stats are fine with a single column; pairwise ones are not.
    model_cfg(1, vec![Mean, Var, Std]).validate().unwrap();
    for s in [Cov, Corr] {
        err(
            model_cfg(1, vec![s]),
            "which compares two features, but the model has one",
        );
    }
    let mut one_pcorr = model_cfg(1, vec![PartialCorr]);
    one_pcorr.precision_prior = Some(1e-6);
    err(
        one_pcorr,
        "which compares two features, but the model has one",
    );

    // partial_corr is read off the tracked precision matrix, which only
    // exists when a prior is configured.
    err(model_cfg(3, vec![PartialCorr]), "precision_prior");
    let mut with_prior = model_cfg(3, vec![PartialCorr]);
    with_prior.precision_prior = Some(1e-6);
    with_prior.validate().unwrap();

    model_cfg(2, vec![Mean, Var, Std, Cov, Corr])
        .validate()
        .unwrap();

    // mahal solves against the regularized co-moments, so it needs the
    // prior too; its quantiles need the score they are quantiles of.
    err(model_cfg(3, vec![Mahal]), "precision_prior");
    let mut mahal = model_cfg(3, vec![Mahal]);
    mahal.precision_prior = Some(1e-6);
    mahal.validate().unwrap();
    let mut mahal1 = model_cfg(1, vec![Mahal]);
    mahal1.precision_prior = Some(1e-6);
    mahal1.validate().unwrap();
    let mut no_mahal = model_cfg(3, vec![Mean]);
    no_mahal.mahal_quantiles = vec![0.99];
    err(no_mahal, "mahal_quantiles needs");
    for bad in [0.0, 1.0, -0.5, 1.5, f64::NAN] {
        let mut c = mahal.clone();
        c.mahal_quantiles = vec![0.5, bad];
        err(c, "mahal_quantiles must each be above 0 and below 1");
    }
    let mut levels = mahal.clone();
    levels.mahal_quantiles = vec![0.5, 0.99];
    levels.validate().unwrap();

    // pca: at most k components, and a cadence that is clock units, 0
    // for every row (task 161).
    let mut too_many = model_cfg(3, vec![Mean]);
    too_many.pca = 4;
    too_many.pca_every = 1.0;
    err(
        too_many,
        "pca must be at most the number of features (3), but it is 4",
    );
    for bad in [-1.0, f64::NAN] {
        for pca in [2, 0] {
            let mut no_cadence = model_cfg(3, vec![Mean]);
            no_cadence.pca = pca;
            no_cadence.pca_every = bad;
            err(no_cadence, "pca_every must be at least 0 clock units");
        }
    }
    let mut ok = model_cfg(3, vec![Mean]);
    ok.pca = 3;
    ok.pca_every = 5.0;
    ok.validate().unwrap();
    let mut off = model_cfg(3, vec![Mean]);
    off.pca = 0;
    off.pca_every = 0.0;
    off.validate().unwrap();
}

/// The most a model's `mahal` of `x` may be off the oracle's
/// `√(δᵀ(C + s·prior·I)⁻¹δ / scale)` with `δ = x − mean` read off the
/// accessors, as these tests write it (docs/PLAN.md task 274), to first
/// order: the model solves by Cholesky ([`crate::oracle::quad_error`]),
/// or `by_precision` reads the precision matrix that `EwCov::precision`
/// forms column by column -- each column `p̂ⱼ` a backward-stable solve,
/// `(A + ΔAⱼ)p̂ⱼ = eⱼ` with `|ΔAⱼ| ≤ E` the Cholesky backward error, so
/// `δᵀ(P̂ − A⁻¹)δ = −Σⱼ (A⁻¹δ)ᵀΔAⱼp̂ⱼδⱼ` is at most `|A⁻¹δ|ᵀE|A⁻¹||δ|`
/// -- then symmetrized (`u`), then `δᵀP̂δ` as `k` row sums and their
/// sum, each sum's rounding read off its run on the model's own `P̂`
/// ([`crate::oracle::running_dot`]); the
/// oracle's LU, a posteriori ([`crate::oracle::quad_form_error`]); and
/// the two `δ`s, which part by the mean's low part (the model's mean is
/// a compensated pair, the accessor its high part) and by three
/// roundings, moving the form by `2|A⁻¹δ|ᵀ|Δδ|`. The quotient by the
/// scale and the root are two roundings a side, and the root halves the
/// form's relative error.
fn mahal_error(cov: &EwCov, x: &[f64], scale: f64, by_precision: bool) -> f64 {
    use crate::oracle::{self, U, abs_form, abs_times};
    let k = cov.k();
    let ridge = cov.precision_prior * cov.precision_scale();
    let m: Vec<f64> = (0..k * k)
        .map(|ij| cov.cov(ij / k, ij % k) + if ij / k == ij % k { ridge } else { 0.0 })
        .collect();
    let dt: Vec<f64> = (0..k).map(|i| x[i] - cov.mean(i)).collect();
    let dd: Vec<f64> = (0..k)
        .map(|i| cov.m_lo[i].abs() + 3.0 * U * dt[i].abs())
        .collect();
    let xs = oracle::solve(&m, &dt);
    let q: f64 = dt.iter().zip(&xs).map(|(a, b)| a * b).sum();
    let e = oracle::cholesky_backward(&m);
    let lib = if by_precision {
        let inv = oracle::inverse(&m);
        let ad: Vec<f64> = dt.iter().map(|v| v.abs()).collect();
        let pa = abs_times(&inv, &ad);
        // The sums' rounding as the model runs them on its own `P̂`.
        let p = cov.precision().unwrap();
        let rows: Vec<(f64, f64)> = (0..k)
            .map(|i| oracle::running_dot((0..k).map(|j| (p[i * k + j], dt[j]))))
            .collect();
        let (_, outer) = oracle::running_dot(dt.iter().zip(&rows).map(|(d, r)| (*d, r.0)));
        let inner: f64 = dt.iter().zip(&rows).map(|(d, r)| d.abs() * r.1).sum();
        let sym = U * abs_form(&ad, &p.iter().map(|v| v.abs()).collect::<Vec<_>>(), &ad);
        abs_form(&xs, &e, &pa) + outer + inner + sym
    } else {
        oracle::quad_error(&m, &dt, &e)
    };
    let own = oracle::quad_form_error(&m, &dt);
    let moved: f64 = 2.0 * xs.iter().zip(&dd).map(|(a, b)| a.abs() * b).sum::<f64>();
    let mahal = (q / scale).sqrt();
    mahal * (0.5 * (lib + own + moved) / q + 4.0 * U)
}

fn mahal_cfg(k: usize, prior: f64) -> EwCovCfg {
    let mut c = model_cfg(k, vec![EwCovStat::Mahal]);
    c.precision_prior = Some(prior);
    c
}

fn step(m: &mut EwCovModel, x: &[f64], first: bool) -> Vec<f64> {
    crate::OnlineModel::step(m, x, &[], if first { 0.0 } else { 1.0 }, 1.0).pred
}

/// Bitwise equality, so NaN slots compare equal to NaN slots.
fn same_bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn mahal_is_the_standardized_distance_on_a_diagonal_covariance() {
    // Two independent columns with variances 4 and 1 (population form
    // under no decay): the distance of (m0 + 2a, m1 + b) is sqrt(a² + b²)
    // up to the prior's `1e-12` on the diagonal.
    let mut m = EwCovModel::new(mahal_cfg(2, 1e-12)).unwrap();
    let pts = [
        [2.0, 1.0],
        [-2.0, -1.0],
        [2.0, -1.0],
        [-2.0, 1.0],
        [2.0, 1.0],
        [-2.0, -1.0],
        [2.0, -1.0],
        [-2.0, 1.0],
    ];
    for (i, p) in pts.iter().enumerate() {
        step(&mut m, p, i == 0);
    }
    // Means 0, var0 = 4, var1 = 1, cov = 0.
    let cov = m.cov();
    assert!(cov.mean(0).abs() < 1e-12 && cov.mean(1).abs() < 1e-12);
    assert!((cov.var(0) - 4.0).abs() < 1e-12 && (cov.var(1) - 1.0).abs() < 1e-12);
    assert!(cov.cov(0, 1).abs() < 1e-12);
    let d = m.mahal(&[6.0, -4.0]); // a = 3, b = -4
    assert!((d - 5.0).abs() < 1e-9, "{d}");
    assert!(m.mahal(&[0.0, 0.0]).abs() < 1e-12);
    // Read off the model, before the row is learned, as slot 0.
    let out = crate::OnlineModel::predict(&m, &[6.0, -4.0], 1.0).pred;
    assert_eq!(out.len(), 1);
    assert_eq!(out[0], d);
}

/// `mahal` beside `partial_corr` is read from the precision matrix the
/// partial correlations factorized, where it used to factorize the same
/// matrix again (task 270, A-2): the distance agrees with the one solved
/// against the row to 4e-15 relative on every row (its largest move is
/// 1.03e-15 on this stream on aarch64, the figure the CHANGELOG quotes;
/// the bound leaves twice that again for x86-64's product kernel, which
/// rounds the factorization's products in another order), and with the
/// definition by the oracle's LU, on a stream with a level, correlated
/// columns, gaps and rows of weight 0, on the unbiased scale and `bias`.
#[test]
fn mahal_beside_partial_corr_is_mahal_alone() {
    for bias in [false, true] {
        let prior = 0.01;
        let mut alone = mahal_cfg(4, prior);
        alone.bias = bias;
        let mut both = alone.clone();
        both.stats = vec![EwCovStat::PartialCorr, EwCovStat::Mahal];
        let (mut a, mut b) = (
            EwCovModel::new(alone).unwrap(),
            EwCovModel::new(both).unwrap(),
        );
        let mut s = 11u64;
        let (mut read, mut largest) = (0, 0.0f64);
        for i in 0..300 {
            let u = lcg(&mut s);
            let x = [
                50.0 + u,
                0.8 * u + 0.3 * lcg(&mut s),
                lcg(&mut s) - 0.5 * u,
                2.0 * lcg(&mut s),
            ];
            let d = if i == 0 {
                0.0
            } else if i % 41 == 0 {
                9.0
            } else {
                1.0
            };
            let w = if i % 13 == 5 { 0.0 } else { 1.0 };
            let pa = crate::OnlineModel::step(&mut a, &x, &[], d, w).pred;
            let pb = crate::OnlineModel::step(&mut b, &x, &[], d, w).pred;
            let (ma, mb) = (pa[0], *pb.last().unwrap());
            if ma.is_nan() {
                assert!(mb.is_nan(), "row {i}: {mb}");
                continue;
            }
            let rel = (ma - mb).abs() / ma.abs().max(f64::MIN_POSITIVE);
            assert!(rel <= 4e-15, "row {i}: {ma} against {mb} ({rel:e})");
            largest = largest.max(rel);
            read += 1;
        }
        assert!(read > 250, "{read} rows read a distance");
        eprintln!("bias = {bias}: largest relative move {largest:e} over {read} rows");
        // The definition, by the oracle's LU.
        let cov = b.cov();
        let ridge = prior * cov.precision_scale();
        let m: Vec<f64> = (0..16)
            .map(|ij| cov.cov(ij / 4, ij % 4) + if ij / 4 == ij % 4 { ridge } else { 0.0 })
            .collect();
        let x = [50.7, -0.4, 1.1, 0.3];
        let delta: Vec<f64> = (0..4).map(|i| x[i] - cov.mean(i)).collect();
        let scale = b.second_moment_scale(cov);
        let want = (crate::oracle::quad_form(&m, &delta) / scale).sqrt();
        let got = *crate::OnlineModel::predict(&b, &x, 0.0)
            .pred
            .last()
            .unwrap();
        // Task 274: the precision matrix's rounding and the oracle's
        // ([`mahal_error`]), where it was `1e-9` relative.
        // Measured on aarch64 (macOS): at most 0.0095 of the bound (2 ulps against an
        // explicit inverse's worst case).
        let bound = mahal_error(cov, &x, scale, true);
        crate::oracle::assert_within(got, want, bound, format_args!("bias {bias}"));
    }
}

#[test]
fn mahal_matches_the_precision_matrix_quadratic_form() {
    // Against the definition, d² = δᵀ(C + s·prior·I)⁻¹δ, the quadratic
    // form by the test oracle's LU (`crate::oracle`). It was read off
    // `precision()`, which solves by the same Cholesky as `mahal`, so an
    // error in that factorization reached both sides alike and passed
    // (docs/PLAN.md task 169).
    let prior = 0.01;
    let mut m = EwCovModel::new(mahal_cfg(3, prior)).unwrap();
    let mut s = 7u64;
    for i in 0..50 {
        let a = lcg(&mut s);
        let x = [a, 0.5 * a + lcg(&mut s), lcg(&mut s) - 0.2 * a];
        step(&mut m, &x, i == 0);
    }
    let cov = m.cov();
    let ridge = prior * cov.precision_scale();
    let a: Vec<f64> = (0..9)
        .map(|ij| cov.cov(ij / 3, ij % 3) + if ij / 3 == ij % 3 { ridge } else { 0.0 })
        .collect();
    let x = [0.7, -0.4, 1.1];
    let delta: Vec<f64> = (0..3).map(|i| x[i] - cov.mean(i)).collect();
    let d2 = crate::oracle::quad_form(&a, &delta);
    // Task 274: the Cholesky solve's rounding and the oracle's
    // ([`mahal_error`]), where it was `1e-9`; the model's own scale,
    // 1 under its `bias`, divides neither side.
    // Measured on aarch64 (macOS): equal to the bit; the result moved by 100u of itself
    // fails it (2.6 times the bound).
    let scale = m.second_moment_scale(cov);
    assert_eq!(scale, 1.0);
    let bound = mahal_error(cov, &x, scale, false);
    crate::oracle::assert_within(m.mahal(&x), d2.sqrt(), bound, "three columns");
    // A single column is the |z| of the row.
    let mut one = EwCovModel::new(mahal_cfg(1, 1e-12)).unwrap();
    for i in 0..20 {
        step(&mut one, &[lcg(&mut s) * 3.0], i == 0);
    }
    let z = (2.5 - one.cov().mean(0)) / one.cov().var(0).sqrt();
    assert!((one.mahal(&[2.5]) - z.abs()).abs() < 1e-9);
}

/// A variance of NaN stays NaN. The floor against rounding was
/// `max(0.0)`, which returns its other argument when one is NaN, so a
/// co-moment a NaN had reached read as a variance of 0, a column with
/// no spread: `ew_cov` emitted 0 for its `var` and `std`, and a
/// diagonal `hmm` or `ew_class` scored on its ridge alone (task 182).
/// A rounding below zero is still 0, and so is `-0.0`, which `f64::max`
/// may order either way.
#[test]
fn a_variance_of_nan_stays_nan() {
    let mut c = EwCov::new(4);
    let mut centred = vec![0.0; 16];
    for (i, v) in [f64::NAN, -1e-18, -0.0, 2.5].into_iter().enumerate() {
        centred[i * 4 + i] = v;
    }
    c.set_moments(&[0.0; 4], &centred, 1.0, None);
    assert!(c.var(0).is_nan(), "{}", c.var(0));
    assert_eq!(c.var(1).to_bits(), 0f64.to_bits());
    assert_eq!(c.var(2).to_bits(), 0f64.to_bits());
    assert_eq!(c.var(3), 2.5);
}

#[test]
fn mahal_quantiles_track_the_scores_seen_so_far() {
    // Each row's own out-of-sample score enters the sketch after it is
    // read, so the quantile field lags by one row and never includes the
    // row it is emitted with; the clock ages the sketch on every row
    // (docs/PLAN.md task 146).
    let mut c = mahal_cfg(2, 1e-6);
    c.mahal_quantiles = vec![0.5];
    let lam = c.decay.factor(1.0);
    let mut m = EwCovModel::new(c).unwrap();
    let mut s = 11u64;
    let mut seen: Vec<(f64, f64)> = Vec::new();
    let mut oracle = crate::EwQuantile::new(&[0.5]).unwrap();
    for i in 0..2000 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let out = step(&mut m, &x, i == 0);
        assert_eq!(out.len(), 2);
        // The field is the estimate *before* this row's score joins.
        assert_eq!(
            out[1].to_bits(),
            oracle.get(0).unwrap_or(f64::NAN).to_bits()
        );
        if i > 0 {
            oracle.age(lam);
            seen.iter_mut().for_each(|e| e.1 *= lam);
        }
        if out[0].is_finite() {
            seen.push((out[0], 1.0));
            oracle.add(out[0], 1.0);
        }
    }
    // Against the definition: the smallest score whose weight, ascending,
    // reaches half the total.
    seen.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let total: f64 = seen.iter().map(|e| e.1).sum();
    let mut cum = 0.0;
    let median = seen
        .iter()
        .find(|e| {
            cum += e.1;
            cum >= 0.5 * total
        })
        .unwrap()
        .0;
    let q = crate::OnlineModel::predict(&m, &[0.0, 0.0], 1.0).pred[1];
    assert_eq!(q.to_bits(), oracle.get(0).unwrap().to_bits());
    assert!((q - median).abs() <= crate::EW_QUANTILE_ALPHA * median * (1.0 + 1e-9));
    // NaN scores (before min_weight) never enter the sketch.
    let mut fresh = EwCovModel::new(m.cfg().clone()).unwrap();
    let out = step(&mut fresh, &[1.0, 2.0], true);
    assert!(out[0].is_nan() && out[1].is_nan());
    assert_eq!(fresh.mahal_q.as_ref().unwrap().get(0), None);
}

/// A state from before schema 21 kept P² markers for its Mahalanobis
/// quantiles: written into msgpack here as that build wrote them, it
/// does not decode. It was read as no sketch and refused by name until
/// the minimum schema passed it (docs/PLAN.md tasks 198 and 194-202). A
/// state with levels and no sketch is refused at `restore`.
#[test]
fn a_state_with_p2_markers_is_refused() {
    fn swap(v: &mut rmpv::Value, with: &rmpv::Value) -> bool {
        match v {
            rmpv::Value::Map(kv) => kv.iter_mut().any(|(k, val)| {
                if k.as_str() == Some("mahal_q") {
                    *val = with.clone();
                    true
                } else {
                    swap(val, with)
                }
            }),
            rmpv::Value::Array(xs) => xs.iter_mut().any(|x| swap(x, with)),
            _ => false,
        }
    }
    let p2 = |p: f64| {
        let f = |xs: [f64; 5]| rmpv::Value::Array(xs.map(rmpv::Value::from).to_vec());
        rmpv::Value::Map(vec![
            ("p".into(), p.into()),
            ("q".into(), f([0.1, 0.2, 0.3, 0.4, 0.5])),
            ("n".into(), f([1.0, 2.0, 3.0, 4.0, 5.0])),
            ("np".into(), f([1.0, 2.0, 3.0, 4.0, 5.0])),
            ("dn".into(), f([0.0, 0.25, 0.5, 0.75, 1.0])),
            ("count".into(), 5.into()),
        ])
    };
    let old = |levels: Vec<f64>| -> Result<crate::State, String> {
        let mut c = mahal_cfg(2, 1e-6);
        c.mahal_quantiles = levels.clone();
        let m = EwCovModel::new(c).unwrap();
        // At this build's schema, the markers' own refusal: a state
        // saved at 20 is refused by its version before it is read
        // (`crate::MIN_SCHEMA_VERSION`).
        let state = crate::OnlineModel::state(&m);
        let bytes = rmp_serde::to_vec_named(&state).unwrap();
        let mut v = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
        let markers = rmpv::Value::Array(levels.iter().map(|&p| p2(p)).collect());
        assert!(swap(&mut v, &markers), "the state has a mahal_q");
        let mut out = Vec::new();
        rmpv::encode::write_value(&mut out, &v).unwrap();
        rmp_serde::from_slice::<crate::State>(&out).map_err(|e| e.to_string())
    };
    assert!(old(vec![0.5, 0.9]).is_err());
    assert!(old(vec![]).is_err());
    let mut c = mahal_cfg(2, 1e-6);
    c.mahal_quantiles = vec![0.5];
    let mut m = EwCovModel::new(c).unwrap();
    m.mahal_q = None;
    let err = <EwCovModel as crate::OnlineModel>::restore(&crate::OnlineModel::state(&m))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("it saved nothing for its mahal_quantiles"),
        "{err}"
    );
}

#[test]
fn pca_of_recovers_a_known_eigenbasis() {
    // [[2, 1], [1, 2]] has eigenpairs (3, (1,1)/√2) and (1, (1,−1)/√2).
    let p = Pca::of(&[2.0, 1.0, 1.0, 2.0], 2, 2, None).unwrap();
    let r = std::f64::consts::FRAC_1_SQRT_2;
    assert!((p.eig[0] - 3.0).abs() < 1e-12 && (p.eig[1] - 1.0).abs() < 1e-12);
    assert_eq!(p.trace, 4.0);
    assert!((p.loadings[0] - r).abs() < 1e-12 && (p.loadings[1] - r).abs() < 1e-12);
    // The tie (|v0| = |v1|) resolves to the first entry positive.
    assert!((p.loadings[2] - r).abs() < 1e-12 && (p.loadings[3] + r).abs() < 1e-12);
    // Only the top component when asked for one.
    let top = Pca::of(&[2.0, 1.0, 1.0, 2.0], 2, 1, None).unwrap();
    assert_eq!(top.r(), 1);
    assert_eq!(top.eig, vec![p.eig[0]]);
    assert_eq!(top.loadings, p.loadings[..2].to_vec());
    // Largest-magnitude entry positive, not the first entry.
    let q = Pca::of(&[1.0, 0.0, 0.0, 5.0], 2, 1, None).unwrap();
    assert!((q.eig[0] - 5.0).abs() < 1e-12);
    assert!(q.loadings[0].abs() < 1e-12 && (q.loadings[1] - 1.0).abs() < 1e-12);
    // Refused: no components, more than k, a non-finite entry, a wrong size.
    assert!(Pca::of(&[1.0, 0.0, 0.0, 1.0], 2, 0, None).is_none());
    assert!(Pca::of(&[1.0, 0.0, 0.0, 1.0], 2, 3, None).is_none());
    assert!(Pca::of(&[1.0, 0.0, 0.0, f64::NAN], 2, 1, None).is_none());
    assert!(Pca::of(&[1.0, 0.0, 0.0], 2, 1, None).is_none());
}

#[test]
fn pca_signs_follow_the_previous_refresh() {
    // The top eigenvector of [[a, -b], [-b, a]] is (1, −1)/√2, and the
    // max-abs rule has a tie there that any perturbation decides. A
    // predecessor along (−1, 1) keeps the new vector on its side.
    let r = std::f64::consts::FRAC_1_SQRT_2;
    let first = Pca::of(&[2.0, -1.0, -1.0, 2.0], 2, 1, None).unwrap();
    assert!((first.loadings[0] - r).abs() < 1e-12 && (first.loadings[1] + r).abs() < 1e-12);
    let prev = Pca {
        eig: vec![3.0],
        trace: 4.0,
        loadings: vec![-r, r],
    };
    let next = Pca::of(&[2.0, -1.0, -1.0, 2.0], 2, 1, Some(&prev)).unwrap();
    assert!((next.loadings[0] + r).abs() < 1e-12 && (next.loadings[1] - r).abs() < 1e-12);
    // Nearly tied magnitudes that trade the lead: max-abs would flip,
    // continuity does not.
    let mut a = Pca::of(&[2.0, -0.999, -0.999, 2.0], 2, 1, None).unwrap();
    for eps in [0.0005, -0.0005, 0.0005, -0.0005] {
        // Tilt the matrix so the lead entry alternates between the two.
        let m = [2.0 + eps, -0.999, -0.999, 2.0 - eps];
        let b = Pca::of(&m, 2, 1, Some(&a)).unwrap();
        let dot = a.loadings[0] * b.loadings[0] + a.loadings[1] * b.loadings[1];
        assert!(dot > 0.99, "dot {dot}");
        a = b;
    }
    // A predecessor with fewer components leaves the extra ones to the
    // max-abs rule.
    let wide = Pca::of(&[2.0, -1.0, -1.0, 2.0], 2, 2, Some(&prev)).unwrap();
    assert!((wide.loadings[0] + r).abs() < 1e-12);
    assert!((wide.loadings[2] - r).abs() < 1e-12 && (wide.loadings[3] - r).abs() < 1e-12);
}

#[test]
fn pca_loadings_are_orthonormal_and_ordered() {
    let k = 5;
    let mut s = 5u64;
    // A random SPD matrix: A = Bᵀ B + small ridge.
    let b: Vec<f64> = (0..k * k).map(|_| lcg(&mut s)).collect();
    let mut a = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            a[i * k + j] = (0..k).map(|l| b[l * k + i] * b[l * k + j]).sum::<f64>()
                + if i == j { 0.01 } else { 0.0 };
        }
    }
    let p = Pca::of(&a, k, k, None).unwrap();
    for j in 1..k {
        assert!(p.eig[j - 1] >= p.eig[j], "descending: {:?}", p.eig);
    }
    assert!((p.eig.iter().sum::<f64>() - p.trace).abs() < 1e-10);
    for i in 0..k {
        for j in 0..k {
            let dot: f64 = (0..k)
                .map(|l| p.loadings[i * k + l] * p.loadings[j * k + l])
                .sum();
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((dot - want).abs() < 1e-10, "v{i}·v{j} = {dot}");
        }
        // A v = λ v.
        for l in 0..k {
            let av: f64 = (0..k).map(|c| a[l * k + c] * p.loadings[i * k + c]).sum();
            assert!((av - p.eig[i] * p.loadings[i * k + l]).abs() < 1e-9);
        }
        let lead = (0..k)
            .max_by(|&x, &y| {
                p.loadings[i * k + x]
                    .abs()
                    .partial_cmp(&p.loadings[i * k + y].abs())
                    .unwrap()
            })
            .unwrap();
        assert!(p.loadings[i * k + lead] > 0.0);
    }
}

/// The window's target floor at its own scale (review 2026-10-05, CB2;
/// the mutation run on `c134b4a` left both products of `terms` alive): a
/// variance at or below `64 eps (g v + ratio v_u)` is 0, one above it is
/// kept, to the bit. With `g = 2`, `ratio = 0.5`, `v_u = 1e6` and
/// `g v = 5e5 + var`, the threshold is `64 eps 1e6`; `g + v` would put it
/// at 0.75 of that, `g / v` at 0.5 and `ratio + v_u` at 1.5, so a variance
/// at 0.88 of it must read 0 and one at 1.27 must be kept.
#[test]
fn the_target_floor_is_sixty_four_epsilons_of_the_terms_it_subtracts() {
    let (g, ratio) = (2.0, 0.5);
    let moments = |var: f64| TargetMoments {
        mean: vec![3.0],
        var: vec![var],
        q: vec![1.0],
        mean_lo: vec![0.0],
    };
    let old = moments(1e6);
    let threshold = 64.0 * f64::EPSILON * 1e6;
    for (share, kept) in [(0.88, false), (1.27, true)] {
        let gv = 5e5 + share * threshold;
        // What the subtraction gives, exactly: `g v` is `gv` (a power of
        // two), `ratio v_u` is 5e5, and their difference is exact.
        let var = gv - 5e5;
        let out = moments(gv / g).truncated(&old, 1.0, &[Some((ratio, g))]);
        let want = if kept { var } else { 0.0 };
        assert_eq!(
            out.var[0].to_bits(),
            want.to_bits(),
            "{share}: {:e}",
            out.var[0]
        );
    }
}

/// The Kish sum's floor is `64 eps` of the live sum, and exclusive: a
/// remainder of exactly that keeps no digit and is 0, and one of twice
/// that is kept. A live sum of 2 puts the floor at `2^-45`, and both
/// subtractions are exact.
#[test]
fn the_target_kish_floor_is_sixty_four_epsilons_of_the_live_sum() {
    let moments = |q: f64| TargetMoments {
        mean: vec![3.0],
        var: vec![1.0],
        q: vec![q],
        mean_lo: vec![0.0],
    };
    for (left, kept) in [(2f64.powi(-45), false), (2f64.powi(-44), true)] {
        let out = moments(2.0).truncated(&moments(2.0 - left), 1.0, &[Some((0.5, 2.0))]);
        let want = if kept { left } else { 0.0 };
        assert_eq!(
            out.q[0].to_bits(),
            want.to_bits(),
            "{left:e}: {:e}",
            out.q[0]
        );
    }
}

/// A target variance that is not a number stays one through the
/// window's truncation, live or in the snapshot: the floor `max(0.0)`
/// read it as 0, a target with no spread inside the window, and the
/// rounding floor after it compares a NaN false (review round 4, CB7;
/// `crate::solve::clamp_rounding`, task 182).
#[test]
fn a_target_variance_that_is_not_a_number_stays_one_in_the_window() {
    let moments = |var: f64| TargetMoments {
        mean: vec![3.0],
        var: vec![var],
        q: vec![1.0],
        mean_lo: vec![0.0],
    };
    for (live, then) in [(f64::NAN, 1.0), (1.0, f64::NAN)] {
        let out = moments(live).truncated(&moments(then), 1.0, &[Some((0.5, 2.0))]);
        assert!(
            out.var[0].is_nan(),
            "live {live}, snapshot {then}: {}",
            out.var[0]
        );
    }
}

/// A cfg read without `max_rows_between_pca` -- a caller's own, or a JSON
/// written before task 161 -- has no row cap, so its components refresh
/// on `pca_every` alone (`no_row_cap`).
#[test]
fn a_cfg_read_without_the_pca_row_cap_has_none() {
    let mut c = model_cfg(3, vec![EwCovStat::Mean]);
    c.pca = 2;
    c.pca_every = 5.0;
    let mut v = serde_json::to_value(&c).unwrap();
    let fields = v.as_object_mut().unwrap();
    assert!(fields.remove("max_rows_between_pca").is_some(), "written");
    let read: EwCovCfg = serde_json::from_value(v).unwrap();
    assert_eq!(read.max_rows_between_pca, u32::MAX);
    assert_eq!(read, c);
}

fn pca_cfg(k: usize, r: usize, every: usize) -> EwCovCfg {
    let mut c = model_cfg(k, vec![EwCovStat::Mean]);
    c.pca = r;
    c.pca_every = every as f64;
    c
}

#[test]
fn lag_corr_fields_and_values_follow_the_lags() {
    let k = 2;
    let mut cfg = model_cfg(k, vec![EwCovStat::Corr, EwCovStat::LagCorr]);
    cfg.decay = crate::Decay::Halflife(30.0);
    cfg.lags = vec![1, 3];
    let names = ["a".to_string(), "b".to_string()];
    let labels = EwCovModel::labels(&names, &cfg.stats, &[], 0, &cfg.lags);
    assert_eq!(
        labels,
        [
            "corr_a_b",
            "lag_corr_l1_a_a",
            "lag_corr_l1_a_b",
            "lag_corr_l1_b_a",
            "lag_corr_l1_b_b",
            "lag_corr_l3_a_a",
            "lag_corr_l3_a_b",
            "lag_corr_l3_b_a",
            "lag_corr_l3_b_b",
        ]
    );
    assert_eq!(cfg.n_outputs(), labels.len());

    // `b` is `a` delayed by one row, so `lag_corr_l1_b_a` -- b now against
    // a one row ago -- goes to 1 and `lag_corr_l1_a_b` does not.
    let mut m = EwCovModel::new(cfg).unwrap();
    let mut s = 4u64;
    let mut prev = 0.0;
    let mut last = Vec::new();
    for i in 0..400 {
        let a = lcg(&mut s);
        let x = [a, prev];
        prev = a;
        last = crate::OnlineModel::step(&mut m, &x, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0).pred;
    }
    // Slots: corr_a_b, then the four at lag 1, then the four at lag 3.
    assert!(last[3] > 0.95, "lag_corr_l1_b_a is {}", last[3]);
    assert!(last[2].abs() < 0.2, "lag_corr_l1_a_b is {}", last[2]);
    // And the auto terms at lag 1 are the series' own autocorrelation,
    // which for white noise is near zero.
    assert!(last[1].abs() < 0.2 && last[4].abs() < 0.2);
}

#[test]
fn clear_lags_empties_the_ring_and_leaves_the_moments() {
    let mut cfg = model_cfg(2, vec![EwCovStat::Corr]);
    cfg.lags = vec![2];
    let mut m = EwCovModel::new(cfg).unwrap();
    let mut s = 7u64;
    for i in 0..30 {
        let x = [lcg(&mut s), lcg(&mut s)];
        crate::OnlineModel::step(&mut m, &x, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let moments = m.cov().clone();
    let lagged = m.lag().unwrap().comoments().to_vec();
    assert_eq!(m.lag().unwrap().depth(), 2);
    crate::OnlineModel::clear_lags(&mut m);
    assert_eq!(m.lag().unwrap().depth(), 0);
    assert_eq!(m.cov(), &moments, "clear_lags is not a reset");
    assert_eq!(m.lag().unwrap().comoments(), &lagged[..]);
}

#[test]
fn lags_do_not_move_the_contemporaneous_moments() {
    let mut plain = EwCovModel::new(model_cfg(3, vec![EwCovStat::Corr])).unwrap();
    let mut cfg = model_cfg(3, vec![EwCovStat::Corr]);
    cfg.lags = vec![1, 2, 5];
    let mut lagged = EwCovModel::new(cfg).unwrap();
    let mut s = 21u64;
    for i in 0..200 {
        let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
        let (d, w) = (if i == 0 { 0.0 } else { 1.0 }, (i % 4) as f64 * 0.5);
        let a = crate::OnlineModel::step(&mut plain, &x, &[], d, w);
        let b = crate::OnlineModel::step(&mut lagged, &x, &[], d, w);
        assert_eq!(a.n_eff, b.n_eff);
        // NaN is not equal to itself, and both are NaN before warmup.
        assert!(
            a.pred
                .iter()
                .zip(&b.pred)
                .all(|(x, y)| x == y || (x.is_nan() && y.is_nan())),
            "{:?} vs {:?}",
            a.pred,
            b.pred
        );
    }
    assert_eq!(plain.cov(), lagged.cov(), "the accumulator is untouched");
}

#[test]
fn a_state_without_lags_resumes_under_a_spec_that_has_them() {
    let mut m = EwCovModel::new(model_cfg(2, vec![EwCovStat::Corr])).unwrap();
    let mut s = 3u64;
    for i in 0..20 {
        let x = [lcg(&mut s), lcg(&mut s)];
        crate::OnlineModel::step(&mut m, &x, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    // A pre-task-48 state: no `lag` field at all. Rewrite the config the
    // way a resumed spec would and restore.
    let mut state = crate::OnlineModel::state(&m);
    if let crate::ModelState::EwCovModel(inner) = &mut state.model {
        inner.cfg.lags = vec![1];
        inner.lag = None;
    }
    let restored = <EwCovModel as crate::OnlineModel>::restore(&state).unwrap();
    let lag = restored.lag().expect("a fresh ring");
    assert_eq!(lag.depth(), 0);
    assert!(lag.comoments().iter().all(|c| *c == 0.0));
    assert_eq!(restored.cov(), m.cov(), "the moments came through");
}

#[test]
fn a_bad_lag_configuration_is_refused_by_name() {
    let mut cfg = model_cfg(2, vec![EwCovStat::LagCorr]);
    assert!(
        EwCovModel::new(cfg.clone())
            .unwrap_err()
            .contains("\"lag_corr\" in stats needs lags")
    );
    cfg.stats = vec![EwCovStat::Corr];
    cfg.lags = vec![2, 1];
    assert!(
        EwCovModel::new(cfg)
            .unwrap_err()
            .contains("strictly increasing")
    );
}

/// `pca_every` counts every row the model is stepped with, rows of
/// weight zero included, as the window's snapshots do (review
/// 2026-09-26, C7): the rule of the cadences that refresh what a model
/// reads out -- the components, the window's snapshots, `coef_every`.
/// The cadences that act on what new rows taught -- `kmeans`'
/// `update_every_rows` and `split_merge_every_rows`, `micro`'s
/// `max_rows_between_prunes` -- count learned rows, since a row of
/// weight zero gives them nothing to act on (the user, 2026-10-06, PLAN
/// task 165). A stream alternating weight 1 and 0
/// under `pca_every = 4` refreshes the components every fourth row, two
/// learned rows apart. The docs said learned rows (review 2026-10-05,
/// CB3; C7 settled `window_every`'s same mismatch by keeping the code
/// and fixing the docs). A refresh is a row after which the counter
/// stands at 0 and did not before.
#[test]
fn a_zero_weight_row_counts_toward_the_pca_cadence() {
    let mut m = EwCovModel::new(pca_cfg(2, 1, 4)).unwrap();
    let mut s = 3u64;
    let mut refreshed = Vec::new();
    for i in 0..80usize {
        let a = lcg(&mut s);
        let x = [a, 2.0 * a + 0.3 * lcg(&mut s)];
        let w = if i % 2 == 0 { 1.0 } else { 0.0 };
        let before = (m.pca.is_some(), m.rows_since_pca);
        crate::OnlineModel::step(&mut m, &x, &[], if i == 0 { 0.0 } else { 1.0 }, w);
        if m.pca.is_some() && m.rows_since_pca == 0 && before != (true, 0) {
            refreshed.push(i);
        }
    }
    let gaps: Vec<usize> = refreshed.windows(2).map(|p| p[1] - p[0]).collect();
    assert!(gaps.len() >= 10, "refreshed at rows {refreshed:?}");
    assert!(
        gaps.iter().all(|&g| g == 4),
        "refreshed at rows {refreshed:?}"
    );
}

/// The regressions' solve schedule, for the components (task 161): a
/// refresh once `pca_every` clock units have passed since the last, or
/// once `max_rows_between_pca` rows have, whichever comes first, and on
/// every row at `pca_every = 0`. Held to that rule written out over the
/// same clock steps: irregular ones, a stretch too close together for
/// the clock, rows at one stamp, and long steps. The first refresh is the
/// first row with `min_weight`, the second here.
#[test]
fn pca_refreshes_on_the_clock_or_the_rows_whichever_comes_first() {
    let steps = [
        0.0, 1.0, 0.5, 2.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 3.0, 0.2, 1.4, 1.0, 0.0, 0.0, 2.6,
        0.3, 0.3,
    ];
    for (every, cap) in [
        (2.5, u32::MAX),
        (f64::INFINITY, 4),
        (2.5, 4),
        (0.0, u32::MAX),
    ] {
        let mut c = pca_cfg(2, 1, 1);
        c.pca_every = every;
        c.max_rows_between_pca = cap;
        let mut m = EwCovModel::new(c).unwrap();
        let mut s = 5u64;
        let mut got = Vec::new();
        for (i, &d) in steps.iter().enumerate() {
            let a = lcg(&mut s);
            let x = [a, 2.0 * a + 0.3 * lcg(&mut s)];
            crate::OnlineModel::step(&mut m, &x, &[], d, 1.0);
            if m.pca.is_some() && m.rows_since_pca == 0 {
                got.push(i);
            }
        }
        let (mut want, mut clock, mut rows) = (Vec::<usize>::new(), 0.0, 0u32);
        for (i, &d) in steps.iter().enumerate() {
            clock += d;
            rows += 1;
            let due = if want.is_empty() {
                i + 1 >= 2
            } else {
                every <= 0.0 || clock >= every || rows >= cap
            };
            if due {
                want.push(i);
                (clock, rows) = (0.0, 0);
            }
        }
        assert_eq!(got, want, "pca_every {every}, max_rows_between_pca {cap}");
        assert!(want.len() >= 5, "{want:?}");
    }
}

/// A save between rows keeps the clock and the rows since the last
/// refresh, so a resumed run refreshes where the unbroken one does, and
/// every output is the same to the bit (task 161).
#[test]
fn the_clock_since_the_last_refresh_survives_a_save() {
    let mut c = pca_cfg(3, 2, 1);
    c.pca_every = 2.5;
    c.max_rows_between_pca = 7;
    let mut s = 9u64;
    let rows: Vec<([f64; 3], f64)> = (0..60)
        .map(|i| {
            let d = if i == 0 {
                0.0
            } else {
                0.1 + f64::from(i % 5) * 0.4
            };
            ([lcg(&mut s), lcg(&mut s), lcg(&mut s)], d)
        })
        .collect();
    let mut a = EwCovModel::new(c.clone()).unwrap();
    let (mut want, mut mid) = (Vec::new(), Vec::new());
    for (x, d) in &rows {
        want.push(crate::OnlineModel::step(&mut a, x, &[], *d, 1.0).pred);
        mid.push(a.since_pca.summed() > 0.0);
    }
    // Each cut where the run is between refreshes, so there is a clock
    // since the last to keep.
    let cuts: Vec<usize> = [5, 17, 33]
        .iter()
        .map(|&from| (from..rows.len()).find(|&i| mid[i - 1]).unwrap())
        .collect();
    for cut in cuts {
        let mut b = EwCovModel::new(c.clone()).unwrap();
        for (i, (x, d)) in rows.iter().enumerate() {
            if i == cut {
                assert!(b.since_pca.summed() > 0.0, "cut {cut}: nothing to keep");
                let bytes = rmp_serde::to_vec(&b).unwrap();
                b = rmp_serde::from_slice(&bytes).unwrap();
            }
            let out = crate::OnlineModel::step(&mut b, x, &[], *d, 1.0).pred;
            assert!(same_bits(&out, &want[i]), "cut {cut}, row {i}");
        }
        assert_eq!(a, b, "cut {cut}");
    }
}

/// A windowed model saved part of a clock spacing after its newest
/// snapshot, through either msgpack encoding, resumes where the
/// unbroken one does, every statistic and `n_eff` to the bit: the
/// spacing, the clock it is measured from and the model's own clock are
/// all in the state (docs/PLAN.md task 162).
#[test]
fn a_window_saved_mid_spacing_resumes_to_the_bit() {
    let mut c = model_cfg(2, vec![EwCovStat::Mean, EwCovStat::Var, EwCovStat::Corr]);
    c.decay = crate::Decay::Halflife(4.0);
    c.min_weight = 0.0;
    c.window = Some(6.0);
    c.window_every = Some(2.5);
    c.max_rows_between_snapshots = Some(7);
    let mut s = 21u64;
    let rows: Vec<([f64; 2], f64)> = (0..90)
        .map(|i| {
            let d = match i {
                0 => 0.0,
                _ if i % 23 == 4 => 0.0,
                _ if i % 31 == 9 => 0.02,
                _ => 0.1 + f64::from(i % 7) * 0.45,
            };
            ([lcg(&mut s), lcg(&mut s)], d)
        })
        .collect();
    let mut a = EwCovModel::new(c.clone()).unwrap();
    let (mut want, mut mid) = (Vec::new(), Vec::new());
    for (x, d) in &rows {
        let out = crate::OnlineModel::step(&mut a, x, &[], *d, 1.0);
        want.push((out.pred, out.n_eff));
        let win = a.win.as_ref().unwrap();
        let since = win.clock - win.snaps.newest().unwrap().clock;
        mid.push(since > 0.0 && since < 2.5);
    }
    let cuts: Vec<usize> = [5, 17, 40, 66]
        .iter()
        .map(|&from| (from..rows.len()).find(|&i| mid[i - 1]).unwrap())
        .collect();
    type Codec = fn(&EwCovModel) -> Vec<u8>;
    let codecs: [(Codec, &str); 2] = [
        (|m| rmp_serde::to_vec(m).unwrap(), "compact"),
        (|m| rmp_serde::to_vec_named(m).unwrap(), "named"),
    ];
    for cut in cuts {
        for (encode, how) in codecs {
            let mut b = EwCovModel::new(c.clone()).unwrap();
            for (i, (x, d)) in rows.iter().enumerate() {
                if i == cut {
                    b = rmp_serde::from_slice(&encode(&b)).unwrap();
                }
                let out = crate::OnlineModel::step(&mut b, x, &[], *d, 1.0);
                let case = format!("cut {cut}, {how}, row {i}");
                assert!(same_bits(&out.pred, &want[i].0), "{case}");
                assert_eq!(out.n_eff.to_bits(), want[i].1.to_bits(), "{case}");
            }
            assert_eq!(a, b, "cut {cut}, {how}");
        }
    }
}

#[test]
fn pca_fields_follow_the_stats_and_are_frozen_between_refreshes() {
    // Outputs: mean_a, mean_b, then pc0_var, pc0_share, pc0_a, pc0_b,
    // pc0_score. With `pca_every = 3` the loadings change only every
    // third row, and a row's score uses the frozen loadings
    // about the live mean.
    let k = 2;
    let mut m = EwCovModel::new(pca_cfg(k, 1, 3)).unwrap();
    assert_eq!(m.cfg().n_outputs(), k + (k + 3));
    let mut s = 3u64;
    let mut last: Option<(Vec<f64>, usize)> = None;
    let mut changes = Vec::new();
    for i in 0..40 {
        let a = lcg(&mut s);
        let x = [a, 2.0 * a + 0.3 * lcg(&mut s)];
        let pre = crate::OnlineModel::predict(&m, &x, 1.0).pred;
        let out = step(&mut m, &x, i == 0);
        assert!(
            same_bits(&pre, &out),
            "predict and step read the same frozen state"
        );
        if !out[2].is_nan() {
            // pc0_score = v·(x − μ) with μ the mean *before* this row.
            let want = out[4] * (x[0] - out[0]) + out[5] * (x[1] - out[1]);
            assert!((out[6] - want).abs() < 1e-12, "row {i}");
            assert!((0.0..=1.0 + 1e-12).contains(&out[3]), "share {}", out[3]);
            let v = out[4..6].to_vec();
            if let Some((prev, at)) = &last {
                if *prev != v {
                    changes.push(i - at);
                    last = Some((v, i));
                }
            } else {
                last = Some((v, i));
            }
        }
    }
    assert!(!changes.is_empty());
    assert!(changes.iter().all(|&c| c == 3), "refresh gaps {changes:?}");
    assert_eq!(m.pca().unwrap().r(), 1);
}

#[test]
fn pca_refresh_counts_rows_so_chunking_cannot_matter() {
    // The same rows through one model and through a clone that was
    // serialized and restored midway give identical outputs, including
    // the refresh cadence and the frozen loadings.
    let mut a = EwCovModel::new(pca_cfg(3, 2, 4)).unwrap();
    let mut s = 9u64;
    let rows: Vec<[f64; 3]> = (0..50)
        .map(|_| [lcg(&mut s), lcg(&mut s), lcg(&mut s)])
        .collect();
    let mut outs = Vec::new();
    for (i, x) in rows.iter().enumerate() {
        outs.push(step(&mut a, x, i == 0));
    }
    let mut b = EwCovModel::new(pca_cfg(3, 2, 4)).unwrap();
    for (i, x) in rows.iter().enumerate() {
        if i == 17 {
            let bytes = rmp_serde::to_vec(&b).unwrap();
            b = rmp_serde::from_slice(&bytes).unwrap();
        }
        let out = step(&mut b, x, i == 0);
        assert!(same_bits(&out, &outs[i]), "row {i}");
    }
    assert_eq!(a, b);
}

#[test]
fn pca_with_every_component_explains_all_the_variance() {
    let k = 3;
    let mut m = EwCovModel::new(pca_cfg(k, k, 1)).unwrap();
    let mut s = 13u64;
    for i in 0..30 {
        step(&mut m, &[lcg(&mut s), lcg(&mut s), lcg(&mut s)], i == 0);
    }
    // Read after the last refresh, which followed the last update.
    let out = crate::OnlineModel::predict(&m, &[0.0; 3], 1.0).pred;
    let shares: f64 = (0..k).map(|j| out[k + j * (k + 3) + 1]).sum();
    assert!((shares - 1.0).abs() < 1e-10, "{shares}");
    let vars: f64 = (0..k).map(|j| out[k + j * (k + 3)]).sum();
    let trace: f64 = (0..k).map(|i| m.cov().cov(i, i)).sum();
    assert!((vars - trace).abs() < 1e-10);
}

#[test]
fn pca_stays_nan_before_min_periods_and_on_a_degenerate_matrix() {
    let mut c = pca_cfg(2, 1, 1);
    c.min_weight = 5.0;
    let mut m = EwCovModel::new(c).unwrap();
    for i in 0..4 {
        let out = step(&mut m, &[i as f64, 1.0], i == 0);
        assert!(out[2..].iter().all(|v| v.is_nan()), "row {i}: {out:?}");
    }
    assert!(m.pca().is_none());
    let out = step(&mut m, &[4.0, 1.0], false);
    assert!(
        out[2..].iter().all(|v| v.is_nan()),
        "read before the first refresh"
    );
    assert!(m.pca().is_some(), "refreshed after the fifth learned row");
    let out = step(&mut m, &[5.0, 1.0], false);
    // Column 1 is constant: all the variance is on column 0.
    assert!((out[3] - 1.0).abs() < 1e-12, "share {}", out[3]);
    assert!((out[4] - 1.0).abs() < 1e-12 && out[5].abs() < 1e-12);
    // A constant stream: trace 0, so the share is NaN rather than 0/0.
    let mut flat = EwCovModel::new(pca_cfg(2, 1, 1)).unwrap();
    let mut out = Vec::new();
    for i in 0..6 {
        out = step(&mut flat, &[1.0, 1.0], i == 0);
    }
    assert_eq!(out[2], 0.0);
    assert!(out[3].is_nan());
}

#[test]
fn n_outputs_and_labels_agree_slot_for_slot() {
    // `n_outputs` sizes the output buffer, `labels` names the fields and
    // `read` fills them: all three must walk the stats in the same order
    // and produce the same count, or a field ends up named after another
    // statistic's value.
    use EwCovStat::*;
    let names: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
    for stats in [
        vec![Mean],
        vec![Var],
        vec![Std],
        vec![Cov],
        vec![Corr],
        vec![Mean, Var, Std, Cov, Corr],
        vec![Corr, Mean],
    ] {
        let mut c = model_cfg(3, stats.clone());
        c.precision_prior = Some(1e-6);
        let labels = EwCovModel::labels(&names, &stats, &[], 0, &[]);
        assert_eq!(c.n_outputs(), labels.len(), "{stats:?}");

        let mut m = EwCovModel::new(c).unwrap();
        let mut s = 31u64;
        for i in 0..40 {
            let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
            crate::OnlineModel::step(&mut m, &x, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let vals = m.read(&m.cov, &[0.0; 3]);
        assert_eq!(vals.len(), labels.len(), "{stats:?}");

        // Spot-check that the label describes the value under it.
        for (label, v) in labels.iter().zip(&vals) {
            if label.starts_with("var_") || label.starts_with("std_") {
                assert!(*v >= 0.0, "{label} = {v}");
            }
            if label.starts_with("corr_") {
                assert!((-1.0..=1.0).contains(v), "{label} = {v}");
            }
        }
    }
}

#[test]
fn labels_name_the_pair_in_column_order() {
    use EwCovStat::*;
    let names: Vec<String> = ["x", "y", "z"].iter().map(|s| s.to_string()).collect();
    assert_eq!(
        EwCovModel::labels(&names, &[Mean], &[], 0, &[]),
        ["mean_x", "mean_y", "mean_z"]
    );
    assert_eq!(
        EwCovModel::labels(&names, &[Var], &[], 0, &[]),
        ["var_x", "var_y", "var_z"]
    );
    assert_eq!(
        EwCovModel::labels(&names, &[Std], &[], 0, &[]),
        ["std_x", "std_y", "std_z"]
    );
    // Upper triangle only, and never a self-pair.
    assert_eq!(
        EwCovModel::labels(&names, &[Cov], &[], 0, &[]),
        ["cov_x_y", "cov_x_z", "cov_y_z"]
    );
    assert_eq!(
        EwCovModel::labels(&names, &[Corr], &[], 0, &[]),
        ["corr_x_y", "corr_x_z", "corr_y_z"]
    );
    assert_eq!(
        EwCovModel::labels(&names, &[PartialCorr], &[], 0, &[]),
        ["partial_corr_x_y", "partial_corr_x_z", "partial_corr_y_z"]
    );
    // Stats concatenate in the order given, not a canonical order.
    assert_eq!(
        EwCovModel::labels(&names, &[Corr, Mean], &[], 0, &[])
            .first()
            .unwrap(),
        "corr_x_y"
    );
}

#[test]
fn read_returns_the_statistic_each_slot_claims() {
    // `read` is the one place the statistics are actually computed rather
    // than accumulated: var vs std vs corr all come off the same moments,
    // so a slot filled from the wrong one is invisible to the accumulator
    // tests. Each is checked against the definition.
    use EwCovStat::*;
    let stats = vec![Mean, Var, Std, Cov, Corr];
    let mut m = EwCovModel::new(model_cfg(2, stats)).unwrap();
    let mut s = 37u64;
    let mut xs = Vec::new();
    for i in 0..80 {
        let a = lcg(&mut s);
        let x = [3.0 + a, 10.0 - 2.0 * a + 0.5 * lcg(&mut s)];
        crate::OnlineModel::step(&mut m, &x, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        xs.push(x.to_vec());
    }
    let v = m.read(&m.cov, &[0.0; 2]);
    let (mean0, mean1) = (v[0], v[1]);
    let (var0, var1) = (v[2], v[3]);
    let (std0, std1) = (v[4], v[5]);
    let cov01 = v[6];
    let corr01 = v[7];

    // Against an unweighted recomputation (half-life is infinite here).
    let n = xs.len() as f64;
    let m0: f64 = xs.iter().map(|x| x[0]).sum::<f64>() / n;
    let m1: f64 = xs.iter().map(|x| x[1]).sum::<f64>() / n;
    let c01: f64 = xs.iter().map(|x| (x[0] - m0) * (x[1] - m1)).sum::<f64>() / n;
    let v0: f64 = xs.iter().map(|x| (x[0] - m0).powi(2)).sum::<f64>() / n;
    let v1: f64 = xs.iter().map(|x| (x[1] - m1).powi(2)).sum::<f64>() / n;

    assert!((mean0 - m0).abs() < 1e-10, "{mean0} vs {m0}");
    assert!((mean1 - m1).abs() < 1e-10, "{mean1} vs {m1}");
    assert!((var0 - v0).abs() < 1e-10, "{var0} vs {v0}");
    assert!((var1 - v1).abs() < 1e-10);
    assert!((std0 - v0.sqrt()).abs() < 1e-10, "std is sqrt(var)");
    assert!((std1 - v1.sqrt()).abs() < 1e-10);
    assert!((cov01 - c01).abs() < 1e-10, "{cov01} vs {c01}");
    assert!(
        (corr01 - c01 / (v0 * v1).sqrt()).abs() < 1e-10,
        "corr is cov / (std*std)"
    );
    // The relationship is strongly negative, so a sign error is visible.
    assert!(corr01 < -0.9, "{corr01}");
}

#[test]
fn accumulate_only_learns_the_same_state_and_emits_nothing() {
    // `stats = []` (docs/ENHANCEMENTS.md E43): no slots, the same
    // accumulator. The two models must agree to the bit, because the
    // whole point is that the state is the product.
    use crate::OnlineModel;
    use EwCovStat::*;
    let mut full = EwCovModel::new(model_cfg(3, vec![Mean, Var, Corr])).unwrap();
    let mut bare = EwCovModel::new(model_cfg(3, vec![])).unwrap();
    assert_eq!(bare.n_outputs(), 0);
    assert!(EwCovModel::labels(&["a".into(), "b".into(), "c".into()], &[], &[], 0, &[]).is_empty());
    let mut s = 11u64;
    for i in 0..200 {
        let x = [lcg(&mut s), 5.0 * lcg(&mut s) + 1.0, lcg(&mut s) - 2.0];
        let (d, w) = (
            if i == 0 { 0.0 } else { 0.7 },
            if i % 7 == 0 { 0.0 } else { 1.5 },
        );
        let sf = full.step(&x, &[], d, w);
        let sb = bare.step(&x, &[], d, w);
        assert!(sb.pred.is_empty(), "no slots");
        assert_eq!(sb.n_eff.to_bits(), sf.n_eff.to_bits());
        assert_eq!(sf.pred.len(), 3 + 3 + 3);
        let (p, pf) = (bare.predict(&x, 0.0), full.predict(&x, 0.0));
        assert!(p.pred.is_empty() && p.n_eff.to_bits() == pf.n_eff.to_bits());
    }
    assert_eq!(bare.cov(), full.cov(), "identical accumulators");
    // The state round-trips and restores into the same accumulator.
    let again = EwCovModel::restore(&bare.state()).unwrap();
    assert_eq!(again.cov(), bare.cov());
    // PCA and Mahalanobis slots stand on their own, without a statistic.
    let mut cfg = model_cfg(3, vec![]);
    cfg.pca = 1;
    cfg.pca_every = 1.0;
    cfg.validate().unwrap();
    assert_eq!(cfg.n_outputs(), 3 + 3);
    let mut cfg = model_cfg(3, vec![]);
    cfg.mahal_quantiles = vec![0.9];
    assert!(cfg.validate().unwrap_err().contains("needs \"mahal\""));
}

#[test]
fn corr_is_nan_not_infinite_when_a_column_is_constant() {
    use EwCovStat::*;
    let mut m = EwCovModel::new(model_cfg(2, vec![Corr, Var])).unwrap();
    let mut s = 41u64;
    for i in 0..40 {
        crate::OnlineModel::step(
            &mut m,
            &[lcg(&mut s), 5.0],
            &[],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    let v = m.read(&m.cov, &[0.0; 2]);
    assert!(v[0].is_nan(), "corr against a constant column: {}", v[0]);
    assert_eq!(v[2], 0.0, "a constant column has exactly zero variance");
}

#[test]
fn matches_direct_computation() {
    let xs: Vec<Vec<f64>> = (0..40)
        .map(|i| {
            let f = i as f64;
            vec![
                f.sin() * 2.0,
                f.cos() - 0.3,
                (f * 0.7).sin() * (f * 0.1).cos(),
            ]
        })
        .collect();
    let lams: Vec<f64> = (0..40).map(|i| 0.9 + 0.0025 * (i % 3) as f64).collect();
    let ws: Vec<f64> = (0..40).map(|i| 0.5 + 0.03 * (i % 7) as f64).collect();

    let mut ew = EwCov::new(3);
    for i in 0..40 {
        ew.update(&xs[i], lams[i], ws[i]);
    }
    let (wsum, m, s) = direct(&xs, &lams, &ws);
    assert!((ew.n_eff() - wsum).abs() < 1e-10);
    for i in 0..3 {
        assert!((ew.mean(i) - m[i]).abs() < 1e-12, "mean {i}");
        for j in 0..3 {
            assert!((ew.raw(i, j) - s[i * 3 + j]).abs() < 1e-12, "raw {i},{j}");
        }
    }
}

#[test]
fn cov_and_var_are_centered() {
    let mut ew = EwCov::new(2);
    for i in 0..1000 {
        let f = i as f64 * (std::f64::consts::TAU / 100.0); // 10 full periods
        ew.update(&[5.0 + f.sin(), -2.0 + f.cos()], 1.0, 1.0);
    }
    // full periods: means equal the offsets, var = 0.5, cov(sin,cos) = 0
    assert!((ew.mean(0) - 5.0).abs() < 0.02);
    assert!((ew.var(0) - 0.5).abs() < 0.02);
    assert!(ew.cov(0, 1).abs() < 0.02);
}

#[test]
fn pure_decay_keeps_means() {
    let mut ew = EwCov::new(1);
    ew.update(&[3.0], 1.0, 1.0);
    ew.update(&[5.0], 1.0, 1.0);
    let m = ew.mean(0);
    let n = ew.n_eff();
    ew.decay(0.5);
    assert_eq!(ew.mean(0), m);
    assert!((ew.n_eff() - 0.5 * n).abs() < 1e-15);
}

#[test]
#[should_panic(expected = "non-negative weight")]
fn a_negative_weight_is_a_caller_error() {
    // The bank rejects negative weights upstream (T-E1), naming the row.
    // Reaching here means that check was bypassed, so this is a
    // `debug_assert`: loud in tests, and a silent no-op in release rather
    // than corrupted state. Only the assertion is testable -- the release
    // path's `if w < 0.0 { return }` is unreachable in a debug build.
    let mut ew = EwCov::new(2);
    ew.update(&[1.0, 2.0], 1.0, -0.5);
}

#[test]
fn an_emptied_accumulator_does_not_divide_by_its_own_zero_weight() {
    // lam = 0 with weight 0 leaves no weight at all; the next update must
    // notice rather than produce NaN.
    let mut empty = EwCov::new(2);
    empty.update(&[1.0, 2.0], 0.0, 0.0);
    assert_eq!(empty.n_eff(), 0.0);
    assert!(empty.mean(0).is_finite() && empty.mean(1).is_finite());
    empty.update(&[3.0, 4.0], 1.0, 1.0);
    assert_eq!(empty.mean(0), 3.0, "the first real row seeds the mean");
    assert_eq!(empty.n_eff(), 1.0);
}

#[test]
fn decay_shrinks_the_weight_and_the_prior_scale_but_not_the_means() {
    // `decay` is the pure-ageing path a skipped row takes. The prior scale
    // must ride along, or the ridge prior stops matching the co-moments it
    // is regularizing.
    let prior = 0.25;
    let mut ew = EwCov::with_precision_prior(2, prior).unwrap();
    let mut s = 109u64;
    for _ in 0..50 {
        ew.update(&[lcg(&mut s), 1.0 + lcg(&mut s)], 0.99, 1.0);
    }
    let (m0, m1) = (ew.mean(0), ew.mean(1));
    let (c01, w, ps) = (ew.cov(0, 1), ew.n_eff(), ew.prior_scale());

    ew.decay(0.5);
    assert_eq!(ew.mean(0), m0, "means are unchanged by pure decay");
    assert_eq!(ew.mean(1), m1);
    assert_eq!(ew.cov(0, 1), c01, "so are the centered co-moments");
    assert!((ew.n_eff() - w * 0.5).abs() < 1e-12, "the weight halves");
    assert!(
        (ew.prior_scale() - ps * 0.5).abs() < 1e-12,
        "so does the prior"
    );
}

#[test]
fn k_reports_the_configured_width() {
    for k in [1usize, 2, 7] {
        assert_eq!(EwCov::new(k).k(), k);
        assert_eq!(EwCov::with_precision_prior(k, 1.0).unwrap().k(), k);
    }
}

#[test]
fn the_model_withholds_statistics_until_min_periods() {
    use crate::OnlineModel;
    let mut c = model_cfg(2, vec![EwCovStat::Mean, EwCovStat::Corr]);
    c.min_weight = 5.0;
    let mut m = EwCovModel::new(c).unwrap();
    let mut s = 113u64;
    for i in 0..10 {
        let step = m.step(
            &[lcg(&mut s), lcg(&mut s)],
            &[],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
        // n_eff is the weight before this row, so it reaches 5 on row 5.
        let want_ready = i >= 5;
        assert_eq!(
            step.pred.iter().all(|v| v.is_finite()),
            want_ready,
            "row {i}: n_eff = {}, min_weight = 5",
            step.n_eff
        );
        assert_eq!(m.n_eff(), step.n_eff + 1.0, "row {i}: n_eff accessor");
    }
}

/// `C + prior·prior_scale·I`, row-major: the matrix `EwCov::precision`
/// inverts, from the co-moments, for the test oracle to invert by faer's
/// partial-pivot LU (`crate::oracle::inverse`), which shares none of the
/// Cholesky `precision()` runs (docs/PLAN.md task 169).
fn regularized(ew: &EwCov, prior: f64, prior_scale: f64) -> Vec<f64> {
    let k = ew.k();
    (0..k * k)
        .map(|ij| {
            let (i, j) = (ij / k, ij % k);
            ew.cov(i, j) + if i == j { prior * prior_scale } else { 0.0 }
        })
        .collect()
}

#[test]
fn the_from_scratch_inverse_really_inverts() {
    // `precision_matches_a_from_scratch_solve` takes its reference from
    // the test oracle (`crate::oracle::inverse`, faer's LU), so that has
    // to be checked against something else -- the definition.
    // A · A⁻¹ = I, where A = C + s·prior·I.
    let (prior, k) = (0.5, 3usize);
    let mut ew = EwCov::with_precision_prior(k, prior).unwrap();
    let mut s = 103u64;
    for _ in 0..80 {
        let x = [lcg(&mut s), 2.0 + lcg(&mut s), 10.0 * lcg(&mut s)];
        ew.update(&x, 0.97, 0.5 + lcg(&mut s).abs());
    }
    let scale = ew.precision_scale();
    let inv = crate::oracle::inverse(&regularized(&ew, prior, scale));
    for i in 0..k {
        for j in 0..k {
            let mut acc = 0.0;
            for m in 0..k {
                let a_im = ew.cov(i, m) + if i == m { prior * scale } else { 0.0 };
                acc += a_im * inv[m * k + j];
            }
            let want = f64::from(i == j);
            assert!(
                (acc - want).abs() < 1e-9,
                "(A A^-1)[{i}][{j}] = {acc}, want {want}"
            );
        }
    }
    // A singular matrix with no prior has no inverse to report: the
    // oracle's is not finite.
    let mut flat = EwCov::new(2);
    for _ in 0..10 {
        flat.update(&[1.0, 2.0], 1.0, 1.0);
    }
    let none = crate::oracle::inverse(&regularized(&flat, 0.0, 1.0));
    assert!(
        none.iter().any(|v| !v.is_finite()),
        "a rank-deficient matrix with no prior cannot be inverted"
    );
}

#[test]
fn the_precision_prior_starts_at_i_over_prior_and_must_be_usable() {
    // Before any data the precision matrix is I/prior exactly.
    let prior = 4.0;
    let ew = EwCov::with_precision_prior(3, prior).unwrap();
    assert_eq!(ew.precision_scale(), 1.0);
    let p = ew.precision().unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let want = if i == j { 1.0 / prior } else { 0.0 };
            assert_eq!(p[i * 3 + j], want, "({i},{j})");
        }
    }
    assert!(ew.has_precision_prior());
    assert!(!EwCov::new(3).has_precision_prior());
    assert!(EwCov::new(3).precision().is_none());
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(EwCov::with_precision_prior(3, bad).is_err(), "prior {bad}");
    }
}

#[test]
fn partial_corr_is_the_textbook_formula_on_the_precision_matrix() {
    // -P_ij / sqrt(P_ii P_jj), against a precision matrix obtained the
    // other way (the test oracle's LU, `crate::oracle`), so the formula
    // and the solve cannot agree by both being wrong.
    let (prior, k) = (1e-4, 4usize);
    let mut ew = EwCov::with_precision_prior(k, prior).unwrap();
    let mut s = 107u64;
    for _ in 0..300 {
        let a = lcg(&mut s);
        let b = lcg(&mut s);
        // A genuine conditional structure, so the values are not all ~0.
        let x = [a, b, a + b + 0.3 * lcg(&mut s), 0.5 * lcg(&mut s)];
        ew.update(&x, 0.99, 1.0);
    }
    let a = regularized(&ew, prior, ew.precision_scale());
    let reference = crate::oracle::inverse(&a);
    let p = ew.precision().unwrap();
    // Task 274: `P̂`'s column `c` is a Cholesky solve, `(A + ΔA_c)x̂_c =
    // e_c` with `|ΔA_c| ≤ E` (`crate::oracle::cholesky_backward`), so
    // `ΔX_rc = −Σ_mn P_rm (ΔA_c)_mn P_nc`, and `P̂_ij` averages `X̂_ij`
    // and `X̂_ji`. Then `ρ = −P_ij/√(P_ii P_jj)` moves by `−ΔP_ij/D +
    // (ρ/2)(ΔP_ii/P_ii + ΔP_jj/P_jj)`, which gathers, to first order,
    // into `Σ_mn (ΔA_j)_mn c_mn + Σ_mn (ΔA_i)_mn c'_mn` with
    // `c_mn = ½P_im P_nj/D − (ρ/2)P_jm P_nj/P_jj` and its mirror: at
    // most `Σ E_mn(|c_mn| + |c'_mn|)`, the signs kept inside each
    // column, where an entry-by-entry bound summed three errors that
    // largely cancel (it sat at 0.006 of the measured error). Beside it
    // the LU's own errors (`crate::oracle::inverse_error`) through the
    // same formula, and each side's two roots, product and quotient,
    // four roundings. It was `1e-6`.
    // Measured on aarch64 (macOS): at most 0.025 of the bound.
    let e = crate::oracle::cholesky_backward(&a);
    let own = crate::oracle::inverse_error(&a);
    let r = &reference;
    for i in 0..k {
        for j in 0..k {
            let (ii, jj, ij) = (i * k + i, j * k + j, i * k + j);
            let want = -r[ij] / (r[ii] * r[jj]).sqrt();
            let got = partial_corr(&p, k, i, j);
            let d = (r[ii] * r[jj]).sqrt();
            let lib: f64 = if i == j {
                0.0
            } else {
                (0..k * k)
                    .map(|mn| {
                        let (m, n) = (mn / k, mn % k);
                        let cj = 0.5 * r[i * k + m] * r[n * k + j] / d
                            - 0.5 * want * r[j * k + m] * r[n * k + j] / r[jj];
                        let ci = 0.5 * r[j * k + m] * r[n * k + i] / d
                            - 0.5 * want * r[i * k + m] * r[n * k + i] / r[ii];
                        e[mn] * (cj.abs() + ci.abs())
                    })
                    .sum()
            };
            let lu = own[ij] / d + 0.5 * want.abs() * (own[ii] / r[ii] + own[jj] / r[jj]);
            let bound = lib + lu + 8.0 * crate::oracle::U * want.abs();
            crate::oracle::assert_within(
                got,
                want.clamp(-1.0, 1.0),
                bound,
                format_args!("pcorr({i},{j})"),
            );
        }
        // A column against itself is -1 by the formula's own definition;
        // callers only ask for i != j, but it must not be NaN.
        assert!(partial_corr(&p, k, i, i).is_finite());
    }
    // Exactly symmetric (the precision matrix is symmetrized), and always
    // a correlation.
    for i in 0..k {
        for j in 0..k {
            let (a, b) = (partial_corr(&p, k, i, j), partial_corr(&p, k, j, i));
            assert_eq!(a, b, "asymmetric at ({i},{j})");
            assert!((-1.0..=1.0).contains(&a), "({i},{j}) = {a}");
        }
    }
}

#[test]
fn precision_matches_a_from_scratch_solve() {
    // The Cholesky solve must equal the test oracle's inversion of the
    // same matrix (faer's partial-pivot LU, `crate::oracle`) at every
    // step, prior scale included.
    let prior = 0.5;
    let mut ew = EwCov::with_precision_prior(3, prior).unwrap();
    let mut state = 7u64;
    let mut lcg = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    for step in 0..60 {
        let x = [lcg(), lcg(), lcg() * 3.0];
        ew.update(&x, 0.97, 0.5 + lcg().abs());
        let a = regularized(&ew, prior, ew.precision_scale());
        let want = crate::oracle::inverse(&a);
        let got = ew.precision().unwrap();
        // Task 274: each column a Cholesky solve, then the two halves
        // averaged (one rounding), against the LU's inverse and its own
        // residual's error (`crate::oracle`), entry by entry. It was
        // `1e-9` relative to `1 + |w|`.
        // Measured on aarch64 (macOS): at most 0.31 of the bound, on a diagonal entry of a
        // 3 x 3 inverse 2.7 ulps off against a worst case of 9.5 (checked against the
        // inverse refined by its exact residual: no term is missing; a factor of order 3
        // has few roundings to lose).
        let (lib, own) = (
            crate::oracle::cholesky_inverse_error(&a),
            crate::oracle::inverse_error(&a),
        );
        for i in 0..3 {
            for j in 0..3 {
                let ij = i * 3 + j;
                let (g, w) = (got[ij], want[ij]);
                let bound = lib[ij].max(lib[j * 3 + i]) + crate::oracle::U * w.abs() + own[ij];
                crate::oracle::assert_within(g, w, bound, format_args!("step {step}, ({i},{j})"));
            }
        }
    }
}

#[test]
fn partial_correlation_detects_a_spurious_link() {
    // x2 = x0 + x1 + noise. Marginally x0 and x2 correlate; controlling for
    // x1 they still do, but x0 and x1 are independent both ways. The
    // interesting case is the reverse: with a common driver, the marginal
    // correlation is high and the partial one is not.
    let mut ew = EwCov::with_precision_prior(3, 1e-6).unwrap();
    let mut state = 11u64;
    let mut lcg = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    for _ in 0..20000 {
        let driver = lcg();
        // two children of one driver: correlated with each other only
        // through it
        let a = driver + 0.1 * lcg();
        let b = driver + 0.1 * lcg();
        ew.update(&[driver, a, b], 1.0, 1.0);
    }
    let marginal = ew.cov(1, 2) / (ew.var(1) * ew.var(2)).sqrt();
    let partial = partial_corr(&ew.precision().unwrap(), 3, 1, 2);
    assert!(
        marginal > 0.9,
        "children should correlate marginally: {marginal}"
    );
    assert!(
        partial.abs() < 0.2,
        "controlling for the driver should remove it: {partial}"
    );
}

#[test]
fn no_precision_prior_by_default() {
    let ew = EwCov::new(2);
    assert!(!ew.has_precision_prior());
    assert!(ew.precision().is_none());
}

/// E48 (task 41): the row scratch is a buffer, not state. A restored
/// accumulator has an empty one and must produce the same bits as the
/// accumulator that wrote it -- which it does, because the scratch is
/// refilled from scratch on every row.
#[test]
fn the_row_scratch_is_not_state() {
    let mut a = EwCov::new(5);
    let mut s = 3u64;
    let row = |st: &mut u64| -> Vec<f64> {
        (0..5)
            .map(|_| {
                *st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((*st >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
            })
            .collect()
    };
    for _ in 0..40 {
        a.update(&row(&mut s), 0.97, 1.5);
    }
    // A round trip leaves the scratch empty and the accumulator equal.
    let bytes = rmp_serde::to_vec_named(&a).unwrap();
    let mut b: EwCov = rmp_serde::from_slice(&bytes).unwrap();
    assert_eq!(a, b, "the scratch must not make two equal states unequal");
    assert!(b.dev.0.is_empty(), "a state file carries no scratch");
    // And both continue to the bit from there.
    let mut s2 = s;
    for _ in 0..40 {
        a.update(&row(&mut s), 0.97, 1.5);
        b.update(&row(&mut s2), 0.97, 1.5);
    }
    assert_eq!(a.comoments(), b.comoments());
    assert_eq!(a.means(), b.means());
    assert_eq!(a.n_eff().to_bits(), b.n_eff().to_bits());
}

#[test]
fn serde_roundtrip() {
    let mut ew = EwCov::new(2);
    ew.update(&[1.0, 2.0], 0.95, 1.3);
    let bytes = rmp_serde::to_vec(&ew).unwrap();
    let back: EwCov = rmp_serde::from_slice(&bytes).unwrap();
    assert_eq!(ew, back);
}

/// Schema 1 stored the Sherman-Morrison inverse under `inv`, the
/// co-moments under `s` and the prior as `inv_prior` / `inv_scale`. It
/// loaded through aliases, with the inverse discarded, until the minimum
/// schema passed it (docs/PLAN.md tasks 198 and 194-202): it does not
/// decode.
#[test]
fn a_schema_1_state_is_refused() {
    let mut want = EwCov::with_precision_prior(2, 0.25).unwrap();
    want.update(&[1.0, 2.0], 0.95, 1.3);
    want.update(&[0.5, 2.5], 0.95, 1.0);
    let v1 = serde_json::json!({
        "k": 2,
        "w_sum": want.w_sum,
        "prior_scale": want.prior_scale,
        "m": want.m,
        "s": want.c,
        "inv": [1.0, 2.0, 3.0, 4.0],
        "inv_prior": 0.25,
        "inv_scale": want.precision_scale,
        "m_lo": want.m_lo,
    });
    assert!(serde_json::from_value::<EwCov>(v1).is_err());
}

#[test]
fn kish_size_of_a_unit_weight_window() {
    // Unit weights on a constant clock: `W -> 1/(1-lam)` and
    // `Q -> 1/(1-lam^2)`, so `W^2/Q -> (1+lam)/(1-lam)`.
    let lam = 0.98;
    let mut ew = EwCov::new(1);
    assert_eq!(ew.n_kish(), None, "no rows, no sample size");
    for i in 0..20_000 {
        ew.update(&[i as f64 % 3.0], lam, 1.0);
    }
    let want = (1.0 + lam) / (1.0 - lam);
    let got = ew.n_kish().unwrap();
    assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    // And it is a *row* count, not a weight: doubling every weight leaves
    // it where it was, while `n_eff` doubles.
    let mut heavy = EwCov::new(1);
    for i in 0..20_000 {
        heavy.update(&[i as f64 % 3.0], lam, 2.0);
    }
    assert!((heavy.n_kish().unwrap() - got).abs() < 1e-9);
    assert!((heavy.n_eff() - 2.0 * ew.n_eff()).abs() < 1e-9);
}

#[test]
fn kish_size_counts_the_rows_that_carry_the_weight() {
    // One heavy row among many light ones is worth about one row.
    let mut ew = EwCov::new(1);
    ew.update(&[1.0], 1.0, 1e6);
    for _ in 0..1_000 {
        ew.update(&[1.0], 1.0, 1.0);
    }
    let n = ew.n_kish().unwrap();
    assert!((1.0..1.01).contains(&n), "{n}");
}

#[test]
fn a_pure_decay_ages_the_squared_weights_too() {
    let mut ew = EwCov::new(1);
    for _ in 0..50 {
        ew.update(&[1.0], 0.9, 1.0);
    }
    let before = ew.n_kish().unwrap();
    ew.decay(0.5);
    // `W` halves and `Q` quarters, so the Kish size is unchanged: aging
    // without data forgets weight, not rows.
    assert!((ew.n_kish().unwrap() - before).abs() < 1e-9);
}

#[test]
fn a_zero_weight_row_is_still_a_decay() {
    let mut a = EwCov::new(1);
    let mut b = EwCov::new(1);
    for _ in 0..10 {
        a.update(&[1.5], 0.9, 1.0);
        b.update(&[1.5], 0.9, 1.0);
    }
    a.update(&[0.0], 0.9, 0.0);
    b.decay(0.9);
    assert_eq!(a.q_sum(), b.q_sum());
    assert_eq!(a.n_eff(), b.n_eff());
}

/// The runs (PLAN task 94) are the rows' business, not the Gram's: a
/// blocked accumulator holds a row the Gram merges later, and a row it
/// skips as a zero-weight one, and its runs are the unblocked one's to
/// the bit whatever it holds. Each run starts at the learned row that
/// changed its value, so a run that covers every learned row started at
/// the first.
#[test]
fn the_runs_follow_the_rows_blocked_or_not() {
    let mut plain = EwCov::new(3);
    let mut blocked = EwCov::new(3);
    blocked.set_block_rows(4);
    for i in 0..60 {
        let x = [
            f64::from(i % 3),
            2.5,
            if i < 40 { f64::from(i) } else { -1.0 },
        ];
        let lam = 0.8 + 0.01 * f64::from(i % 7);
        match i % 6 {
            1 => {
                plain.skip(&x, lam);
                blocked.skip(&x, lam);
            }
            4 => {
                plain.decay(lam);
                blocked.decay(lam);
            }
            5 => {
                plain.update(&[9.0, 9.0, 9.0], lam, 0.0);
                blocked.update(&[9.0, 9.0, 9.0], lam, 0.0);
            }
            _ => {
                let w = 0.5 + f64::from(i % 4);
                plain.update(&x, lam, w);
                blocked.update(&x, lam, w);
            }
        }
        assert_eq!(plain.runs, blocked.runs, "row {i}");
    }
    // Feature 1 held 2.5 on every learned row: its run started at the
    // first.
    assert_eq!(plain.held_from(1, 1), Some(2.5));
    let run = |i: usize| plain.runs.get(i).unwrap();
    assert_eq!(run(1).1, 1);
    // Feature 0 cycles, so its run is the last learned row alone; the
    // zero-weight rows of 9s ended nothing and counted for nothing.
    let last = plain.rows_learned();
    assert_eq!(plain.held_from(0, last - 1), None);
    assert_eq!(plain.held_from(0, last), Some(run(0).0));
    assert_ne!(run(0).0, 9.0);
    // Feature 2 has held -1 since row 40, a learned row in the middle.
    assert_eq!(run(2).0, -1.0);
    assert!(run(2).1 > 1 && run(2).1 < last);
    assert_eq!(plain.held_from(2, run(2).1), Some(-1.0));
    assert_eq!(plain.held_from(2, run(2).1 - 1), None);
}

/// New moments stand for rows the runs never saw -- `gaps`' blend with
/// the slow twin -- so after `set_moments` no run covers any weight, and
/// the next learned row starts each again.
/// A feature that moves and then holds, at a level of 1e8, per row and
/// in blocks of every size: its mean, a pair ([`crate::comp`]), reaches
/// the value exactly, and its variance and its covariance with a feature
/// that moves decay with the history, as exact arithmetic has them
/// (docs/PLAN.md task 101). A plain mean over the same rows stops a few
/// rounding steps short, and left both on the gap.
#[test]
fn a_held_slot_reaches_its_value_per_row_and_in_blocks() {
    for block in [0, 1, 4, 16] {
        let mut cov = EwCov::new(2);
        cov.set_block_rows(block);
        let (level, held) = (1e8, 1e8 + 0.37);
        let lam = 0.5f64.powf(1.0 / 20.0);
        let mut s = 3u64;
        let (mut at_stop, mut plain, mut w_sum) = (0.0, 0.0, 0.0);
        for i in 0..3300 {
            let x = [
                lcg(&mut s),
                if i < 300 { level + lcg(&mut s) } else { held },
            ];
            cov.update(&x, lam, 1.0);
            w_sum = lam * w_sum + 1.0;
            plain += (x[1] - plain) / w_sum;
            if i == 299 {
                cov.flush();
                at_stop = cov.var(1);
            }
        }
        cov.flush();
        assert_ne!(plain, held, "the plain mean reaches it; no case");
        assert_eq!(cov.mean(1), held, "block {block}");
        assert!(
            cov.var(1) <= 1e-30 * at_stop,
            "block {block}: {:e}",
            cov.var(1)
        );
        let corr = cov.cov(0, 1) / (cov.var(0) * cov.var(1)).sqrt();
        assert!(
            corr.is_nan() || corr.abs() <= 1e-10,
            "block {block}: correlation {corr}"
        );
    }
}

/// A skipped row under a capped gap (`lam = 0`) empties a blocked
/// accumulator as it empties a plain one: nothing is buffered where
/// there is no weight to hold it against, and `decay` is the answer.
#[test]
fn a_capped_gap_on_a_skipped_row_empties_a_blocked_accumulator() {
    let mut ew = EwCov::new(2);
    ew.set_block_rows(4);
    for i in 0..3 {
        ew.update(&[f64::from(i), 1.0], 0.9, 1.0);
    }
    assert!(ew.n_eff() > 0.0);
    ew.skip(&[7.0, 7.0], 0.0);
    assert_eq!(ew.n_eff(), 0.0, "a gap past the cap forgets everything");
    assert!(!ew.has_pending());
}

/// The residue of a block's mean is folded in with its sign, and into
/// the pair on its own. At a level of 1e8 and a spread of 1, the block
/// mean pass one computes rounds to the level's step, 1.5e-8, and the
/// residue is that rounding, taken exactly (each `x − m_B` is exact,
/// the two being within a factor of two). Read the wrong way, or added
/// to a first block's `m_B − 0` before the pair sees it, it leaves the
/// merged pair a step off exact arithmetic, where the per-row pair is
/// within 1e-12 of it.
#[test]
fn the_residue_corrects_the_block_mean() {
    let (k, block, level) = (2usize, 8usize, 1e8);
    let mut per_row = EwCov::new(k);
    let mut blocked = EwCov::new(k);
    blocked.set_block_rows(block);
    let mut s = 7u64;
    let mut flushed = 0;
    for i in 0..10 * block {
        let x: Vec<f64> = (0..k).map(|_| level + lcg(&mut s)).collect();
        let w = 1.0 + (i % 3) as f64 / 2.0;
        per_row.update(&x, 0.9, w);
        blocked.update(&x, 0.9, w);
        if blocked.has_pending() {
            continue;
        }
        flushed += 1;
        for j in 0..k {
            let gap = crate::comp::dev(per_row.mean(j), blocked.mean(j), blocked.mean_lo(j))
                + per_row.mean_lo(j);
            assert!(
                gap.abs() <= 1e-12,
                "row {i}, slot {j}: the blocked pair is {gap:e} from the per-row pair"
            );
        }
        // The residue enters the co-moments' corrections too, where a
        // wrong sign is 1e-9 of them.
        for (b, p) in blocked
            .comoments()
            .to_full()
            .iter()
            .zip(per_row.comoments().to_full())
        {
            assert!(
                (b - p).abs() <= 1e-12 * (1.0 + p.abs()),
                "row {i}: co-moment {b} against the per-row {p}"
            );
        }
    }
    assert_eq!(flushed, 10, "every block flushed");
}

/// Task 94's zero reaches `ew_cov`'s own window (review 2026-09-25):
/// a feature held over the window has no spread there, so its variance
/// is exactly 0 and its covariances with the others exactly 0, where the
/// subtraction left a remainder that read as a correlation.
#[test]
fn ew_covs_window_reads_a_held_feature_as_no_spread() {
    let mut m = EwCovModel::new(EwCovCfg {
        bias: true,
        n_features: 2,
        decay: crate::Decay::Halflife(10.0),
        stats: vec![EwCovStat::Var, EwCovStat::Corr],
        min_weight: 0.0,
        precision_prior: None,
        mahal_quantiles: Vec::new(),
        pca: 0,
        pca_every: 0.0,
        max_rows_between_pca: u32::MAX,
        lags: Vec::new(),
        window: Some(30.0),
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap();
    let mut s = 9u64;
    for i in 0..120 {
        let x1 = if i < 60 {
            1e6 + lcg(&mut s)
        } else {
            1e6 + 0.37
        };
        crate::OnlineModel::step(
            &mut m,
            &[lcg(&mut s), x1],
            &[],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    let w = m.windowed_cov();
    assert_eq!(w.var(1), 0.0);
    assert_eq!(w.cov(0, 1), 0.0);
    assert!(w.var(0) > 0.0);
}

/// A window whose rows the live accumulator no longer resolves reads the
/// feature as having no spread, where it read the rounding (review
/// 2026-09-27, G5, from the contract's proptest). Rows at `1e100` of
/// weight 1 before the window, then one of weight `1e100` inside it at
/// an ordinary value: the live co-moments are about `1e100`, the
/// window's own contribution is below their last digit, and the
/// subtraction returned noise near `1e84`, which the lasso fitted to a
/// prediction of `-inf`. A variance smaller than the rounding of the
/// terms it is formed from carries no digits of it. An ordinary window
/// is untouched: the held-feature test above and every window oracle.
#[test]
fn a_window_the_live_state_cannot_resolve_reads_as_no_spread() {
    let mut m = EwCovModel::new(EwCovCfg {
        bias: true,
        n_features: 2,
        decay: crate::Decay::Halflife(20.0),
        stats: vec![EwCovStat::Var, EwCovStat::Corr],
        min_weight: 0.0,
        precision_prior: None,
        mahal_quantiles: Vec::new(),
        pca: 0,
        pca_every: 0.0,
        max_rows_between_pca: u32::MAX,
        lags: Vec::new(),
        window: Some(7.0),
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap();
    // `(x, clock step, weight)`; the step of 12.9 puts every row at
    // `1e100` outside the window of 7.
    let rows: [([f64; 2], f64, f64); 12] = [
        ([0.0, 0.0], 0.0, 1.0),
        ([0.0, 1e100], 1.0, 1.0),
        ([0.0, 0.0], 1.0, 0.7),
        ([0.0, 0.0], 1.0, 1.0),
        ([0.0, 0.0], 1.0, 1.0),
        ([0.0, 0.0], 1.0, 1.0),
        ([1e100, -7.7e99], 1.0, 1.0),
        ([0.0, 0.0], 12.9, 0.01),
        ([0.0, 0.0], 1.0, 1.0),
        ([0.0, 0.0], 1.0, 1.0),
        ([-2.98, 1.02], 1.0, 1e100),
        ([0.0, 0.0], 1.0, 1.0),
    ];
    for (x, d, w) in rows {
        crate::OnlineModel::step(&mut m, &x, &[], d, w);
    }
    let w = m.windowed_cov();
    for i in 0..2 {
        // The true spread inside the window is about 9e-100 of the
        // weight: nothing a double resolves against 1e100.
        assert!(w.var(i) < 1e-50, "var {i}: {:e}", w.var(i));
    }
    assert!(w.cov(0, 1).abs() < 1e-50, "{:e}", w.cov(0, 1));
}

/// Every learned row counts for the runs, and a row of weight 0 does
/// not. A row of any other weight is a learned row, however light next
/// to the live weight: the window alone says what is nothing, by its
/// own weight (review 2026-10-05, CE3). A row at `EMPTY_FRACTION` of the
/// live weight or less counted for no run, so a window of such rows,
/// together above the fraction, read every feature as held at the last
/// heavy row's value.
#[test]
fn learned_rows_count_for_the_runs_and_rows_of_no_weight_do_not() {
    let mut ew = EwCov::new(1);
    for i in 0..20 {
        ew.update(&[5.0], 0.9, if i % 5 == 4 { 0.0 } else { 1.0 });
    }
    assert_eq!(ew.rows_learned(), 16);
    let mut ew = EwCov::new(1);
    ew.update(&[5.0], 1.0, 1.0);
    ew.update(&[7.0], 1.0, crate::window::EMPTY_FRACTION);
    assert_eq!(ew.rows_learned(), 2, "the row at the fraction counts");
    assert_eq!(ew.held_from(0, 2), Some(7.0), "and started a run");
    ew.update(&[5.0], 1.0, 1.0);
    assert_eq!(ew.held_from(0, 2), None, "which the next row ended");
    ew.update(&[5.0], 1.0, 1e-300);
    assert_eq!(ew.rows_learned(), 4, "a row far below the fraction");
    assert_eq!(ew.held_from(0, 3), Some(5.0), "extends the run it carries");
    ew.update(&[9.0], 1.0, 0.0);
    assert_eq!(ew.rows_learned(), 4, "a row of weight 0 does not count");
    assert_eq!(ew.held_from(0, 3), Some(5.0), "nor end a run");
}

/// A window whose rows are each lighter than `EMPTY_FRACTION` of the
/// live weight, and together heavier, reads their moments (review
/// 2026-10-05, CE3, as `marginal` does). A thousand unit rows, then
/// twelve at `1e-9`, a window of 10 clock units and no decay: the runs
/// counted none of the light rows, so the window read each feature as
/// held at the last heavy row's value, with no spread. Against the
/// eleven rows inside, to the digits a window of `1.1e-11` of the weight
/// leaves the subtraction, about five.
/// Kish's size inside `ew_cov`'s window, `W_R² / Q_R` from
/// `crate::truncated`'s remainders: where `Q_R` keeps digits, the rows
/// inside the window's, to the digits kept; where the subtraction of the
/// Kish sums leaves none -- `Q_R` within `64 ε` of the sum it came from
/// -- no size, as `marginal`'s and the regressions' Grams read it. It
/// read 10.75 against 11 with the window's rows at 1e-6 (review
/// 2026-10-05, CE6b). A thousand unit rows, then twelve light ones, a
/// window of 10 clock units and no decay.
#[test]
fn kish_size_inside_a_window_is_the_rows_inside_or_nothing() {
    for (light, digits) in [
        (1e-3, true),
        (1e-5, true),
        (1e-6, false),
        (1e-7, false),
        (1e-9, false),
    ] {
        let mut c = model_cfg(2, vec![EwCovStat::Mean]);
        c.decay = crate::Decay::Halflife(f64::INFINITY);
        c.window = Some(10.0);
        let mut m = EwCovModel::new(c).unwrap();
        // Eleven rows inside, the row exactly a window old among them.
        crate::OnlineModel::set_window_closed(&mut m, crate::WindowClosed::Both);
        let mut s = 5u64;
        for i in 0..1012 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let w = if i < 1000 { 1.0 } else { light };
            let d = if i == 0 { 0.0 } else { 1.0 };
            crate::OnlineModel::step(&mut m, &x, &[], d, w);
        }
        let got = m.windowed_cov().n_kish();
        if digits {
            // Eleven rows of one weight: `(11 w)² / (11 w²) = 11`.
            let got = got.unwrap_or_else(|| panic!("{light}: no size"));
            assert!((got - 11.0).abs() <= 1e-3 * 11.0, "{light}: {got}");
        } else {
            assert_eq!(got, None, "{light}");
        }
    }
}

#[test]
fn a_window_of_light_rows_reads_their_moments() {
    let mut c = model_cfg(2, vec![EwCovStat::Mean]);
    c.decay = crate::Decay::Halflife(f64::INFINITY);
    c.window = Some(10.0);
    let mut m = EwCovModel::new(c).unwrap();
    // Eleven rows inside, the row exactly a window old among them.
    crate::OnlineModel::set_window_closed(&mut m, crate::WindowClosed::Both);
    let mut s = 5u64;
    let mut rows = Vec::new();
    for i in 0..1012 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let w = if i < 1000 { 1.0 } else { 1e-9 };
        let d = if i == 0 { 0.0 } else { 1.0 };
        crate::OnlineModel::step(&mut m, &x, &[], d, w);
        rows.push(x);
    }
    let inside = &rows[rows.len() - 11..];
    let n = inside.len() as f64;
    let mean = |i: usize| inside.iter().map(|r| r[i]).sum::<f64>() / n;
    let cov = |i: usize, j: usize| {
        let (mi, mj) = (mean(i), mean(j));
        inside
            .iter()
            .map(|r| (r[i] - mi) * (r[j] - mj))
            .sum::<f64>()
            / n
    };
    let w = m.windowed_cov();
    assert!((w.n_eff() - 11e-9).abs() <= 1e-3 * 11e-9, "{:e}", w.n_eff());
    for i in 0..2 {
        assert!(
            (w.mean(i) - mean(i)).abs() <= 1e-3,
            "mean {i}: {}",
            w.mean(i)
        );
        let v = cov(i, i);
        assert!(
            (w.var(i) - v).abs() <= 1e-3 * v,
            "var {i}: {} against {v}",
            w.var(i)
        );
    }
    let (c01, scale) = (cov(0, 1), (cov(0, 0) * cov(1, 1)).sqrt());
    assert!(
        (w.cov(0, 1) - c01).abs() <= 1e-3 * scale,
        "{} against {c01}",
        w.cov(0, 1)
    );
}

#[test]
fn new_moments_forget_the_runs() {
    let mut ew = EwCov::new(2);
    for _ in 0..10 {
        ew.update(&[1.0, 2.0], 0.9, 1.0);
    }
    assert_eq!(ew.held_from(0, 1), Some(1.0));
    let (m, c) = (ew.means().to_vec(), ew.comoments().to_full());
    ew.set_moments(&m, &c, ew.n_eff(), ew.q_sum());
    assert!(!ew.runs.is_known(2), "forgotten");
    assert_eq!(ew.held_from(0, u64::MAX), None);
    ew.update(&[1.0, 3.0], 0.9, 2.0);
    assert_eq!(
        (ew.runs.get(0), ew.runs.get(1)),
        (Some((1.0, 11)), Some((3.0, 11))),
        "started again at the eleventh learned row, the value the same or not"
    );
}

/// A state written before schema 16 has no runs (the named encoding
/// defaults them to empty): no feature is held over anything, and the
/// next learned row starts every run at its own weight.
#[test]
fn a_state_without_runs_starts_them_at_its_next_row() {
    let mut ew = EwCov::new(2);
    for i in 0..20 {
        ew.update(&[1.0, f64::from(i)], 0.9, 1.0);
    }
    let mut old = serde_json::to_value(&ew).unwrap();
    let map = old.as_object_mut().unwrap();
    assert!(map.remove("runs").is_some());
    let mut back: EwCov = serde_json::from_value(old).unwrap();
    assert!(!back.runs.is_known(2));
    assert_eq!(back.held_from(0, u64::MAX), None, "nothing is known");
    back.update(&[1.0, 5.0], 0.9, 0.75);
    assert_eq!(
        (back.runs.get(0), back.runs.get(1)),
        (Some((1.0, 21)), Some((5.0, 21))),
        "the twenty-first learned row starts each"
    );
    // A state with one half of the runs and not the other knows no more.
    for key in ["x", "start"] {
        let mut half = serde_json::to_value(&ew).unwrap();
        half["runs"].as_object_mut().unwrap().remove(key);
        let mut back: EwCov = serde_json::from_value(half).unwrap();
        assert_eq!(
            back.held_from(0, u64::MAX),
            None,
            "{key} missing: nothing is known"
        );
        back.update(&[1.0, 5.0], 0.9, 0.75);
        assert_eq!(back.runs.get(1), Some((5.0, 21)), "{key} missing");
    }
}

#[test]
fn target_moments_match_an_ew_cov_over_the_column() {
    // `TargetMoments::learn` takes the `a`/`b` its caller computed from
    // the same per-target weight, so its variance is bit-identical to an
    // `EwCov` over that one column.
    let mut tm = TargetMoments::new(1);
    let mut ew = EwCov::new(1);
    let mut w_t = 0.0;
    for i in 0..500 {
        let y = 100.0 + (i as f64 * 0.37).sin();
        let (lam, w) = (0.97, 0.5 + (i % 4) as f64);
        let w_new = lam * w_t + w;
        let (a, b) = (lam * w_t / w_new, w / w_new);
        tm.learn(0, y, a, b, lam, w);
        w_t = w_new;
        ew.update(&[y], lam, w);
    }
    assert_eq!(tm.means()[0], ew.mean(0), "mean");
    assert_eq!(tm.vars()[0], ew.cov(0, 0), "variance");
    assert_eq!(tm.q()[0], ew.q_sum().unwrap(), "Sum w^2");
    assert_eq!(tm.n_kish(&[w_t])[0], ew.n_kish(), "Kish n");
}

/// At a level of 1e8 as well as 0 (review 2026-09-12, C16; docs/PLAN.md
/// task 112): the blend takes centred moments, so blending with a copy
/// changes nothing at any level, which a blend through the raw second
/// moment could not keep at 1e8.
#[test]
fn blending_target_moments_with_a_copy_is_the_identity() {
    for level in [0.0, 1e8] {
        blending_a_copy_at(level);
    }
}

fn blending_a_copy_at(level: f64) {
    let mut tm = TargetMoments::new(1);
    let mut w_t = 0.0;
    for i in 0..100 {
        let y = level + (i as f64 * 0.11).cos();
        let (lam, w) = (0.95, 1.0);
        let w_new = lam * w_t + w;
        tm.learn(0, y, lam * w_t / w_new, w / w_new, lam, w);
        w_t = w_new;
    }
    let twin = tm.clone();
    let want = tm.clone();
    tm.blend(&twin, 0, 0.5, 0.5);
    assert!((tm.means()[0] - want.means()[0]).abs() <= 1e-12 * (1.0 + level));
    // The variance is about 0.5 at any level, and must stay it.
    assert!(
        (tm.vars()[0] - want.vars()[0]).abs() < 1e-12,
        "level {level}: {} vs {}",
        tm.vars()[0],
        want.vars()[0]
    );
    // `Q` too: a union of the two row sets would report the blend as
    // twice as informative as the state it blended with itself.
    assert!((tm.q()[0] - want.q()[0]).abs() < 1e-12);
}

/// Runs are a window's (docs/PLAN.md task 128): `ew_cov` keeps them under
/// a window alone, and an accumulator built `without_runs` keeps none,
/// whatever it learns.
#[test]
fn only_a_windowed_ew_cov_keeps_runs() {
    let mut cfg = model_cfg(2, vec![EwCovStat::Mean]);
    let plain = EwCovModel::new(cfg.clone()).unwrap();
    cfg.window = Some(10.0);
    let windowed = EwCovModel::new(cfg).unwrap();
    assert!(!plain.cov.keeps_runs());
    assert!(windowed.cov.keeps_runs());
    let mut c = EwCov::new(2).without_runs();
    for _ in 0..3 {
        c.update(&[1.0, 2.0], 0.9, 1.0);
    }
    assert!(!c.keeps_runs());
    assert_eq!(c.held_from(0, 1), None, "no run is kept, so none is read");
    let mut on = EwCov::new(2);
    on.update(&[1.0, 2.0], 0.9, 1.0);
    assert_eq!(on.held_from(0, 1), Some(1.0));
}

// --- the weekly pass's survivors (docs/PLAN.md task 158) --------------

/// `from_moments` takes `k` means and `k²` co-moments, each checked on
/// its own, at `k = 3` where `k²` is not `2k`; and a weight that is
/// finite and not negative, 0 included.
#[test]
fn from_moments_checks_each_shape_and_the_weight() {
    let (m3, c9) = (vec![0.0; 3], vec![0.0; 9]);
    assert!(EwCov::from_moments(3, 1.0, &m3, &c9, 1.0).is_ok());
    assert!(
        EwCov::from_moments(3, 1.0, &m3[..2], &c9, 1.0).is_err(),
        "means short"
    );
    assert!(
        EwCov::from_moments(3, 1.0, &m3, &c9[..6], 1.0).is_err(),
        "co-moments 2k"
    );
    assert!(
        EwCov::from_moments(3, 0.0, &m3, &c9, 1.0).is_ok(),
        "no weight yet"
    );
    for bad in [-1.0, f64::INFINITY, f64::NAN] {
        assert!(
            EwCov::from_moments(3, bad, &m3, &c9, 1.0).is_err(),
            "w_sum {bad}"
        );
    }
}

/// `has_shape` holds each part to `k` on its own.
#[test]
fn has_shape_checks_each_part() {
    let ok = EwCov::new(3);
    assert!(ok.has_shape(3) && !ok.has_shape(2));
    let mut m = ok.clone();
    m.m.pop();
    assert!(!m.has_shape(3), "a mean short");
    let mut c = ok.clone();
    c.c.pop();
    assert!(!c.has_shape(3), "a co-moment short");
}

/// A target that has not seen a weighted row has no Kish count; a blend
/// is the centred mixture `a·v + b·v' + a·b·(m − m')²` of the two
/// variances and the mixture of the means; and a partial correlation
/// with no precision on a diagonal is NaN, not ±1.
#[test]
fn target_moments_kish_and_blend_and_a_partial_corr_without_precision() {
    let tm = TargetMoments::new(2);
    assert_eq!(tm.n_kish(&[0.0, 0.0]), vec![None, None]);
    let mut a = TargetMoments::new(1);
    (a.mean[0], a.var[0]) = (1.0, 2.0);
    let mut b = TargetMoments::new(1);
    (b.mean[0], b.var[0]) = (4.0, 3.0);
    a.blend(&b, 0, 0.25, 0.75);
    assert_eq!(a.var[0], 0.25 * 2.0 + 0.75 * 3.0 + 0.25 * 0.75 * 9.0);
    assert_eq!(a.mean[0], 3.25);
    assert!(partial_corr(&[0.0, 0.5, 0.5, 2.0], 2, 0, 1).is_nan());
}

/// `window_size` is finite and positive: 0, a negative, `inf` and NaN
/// are each refused.
#[test]
fn an_ew_cov_window_must_be_finite_and_positive() {
    for bad in [0.0, -3.0, f64::INFINITY, f64::NAN] {
        let mut c = model_cfg(2, vec![EwCovStat::Mean]);
        c.window = Some(bad);
        let err = c.validate().unwrap_err();
        assert!(err.contains("window_size"), "{bad}: {err}");
    }
}

/// Each component's sign is set for continuity with the previous
/// refresh's same component, `v_new · v_old >= 0`: on `diag(1, 2, 3)`
/// with the previous second component `−e₂`, the second stays `−e₂`;
/// on a 2x2 matrix whose top vector is `(0.8, 0.6)`, a previous
/// `(0.7, −0.714)` (dot 0.13) keeps it `(0.8, 0.6)`. A previous loading
/// exactly orthogonal (here zero) falls to the rule for none: the
/// largest-magnitude entry positive, on a matrix whose raw eigenvector
/// faer gives with that entry negative (`crate::oracle::sym_eigen`, the
/// eigensolver `Pca::of` runs, its signs untouched). And eigenvalues
/// that overflow leave no components.
#[test]
fn pca_signs_follow_the_previous_component() {
    let c = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
    let prev = Pca {
        eig: vec![3.0, 2.0],
        trace: 6.0,
        loadings: vec![0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
    };
    let p = Pca::of(&c, 3, 2, Some(&prev)).unwrap();
    assert_eq!(p.loadings, vec![0.0, 0.0, 1.0, 0.0, -1.0, 0.0]);

    let c = [2.28, 0.96, 0.96, 1.72];
    let prev = Pca {
        eig: vec![3.0],
        trace: 4.0,
        loadings: vec![0.7, -0.714],
    };
    let p = Pca::of(&c, 2, 1, Some(&prev)).unwrap();
    assert!(
        (p.loadings[0] - 0.8).abs() < 1e-12 && (p.loadings[1] - 0.6).abs() < 1e-12,
        "{p:?}"
    );

    let c = [
        1.9692192435623304,
        -1.0189929071772608,
        -1.678794952215703,
        0.41393543745829764,
        -1.0189929071772608,
        1.1803220149294875,
        1.2278042677158563,
        -0.45118386359032686,
        -1.678794952215703,
        1.2278042677158563,
        1.9379537610346815,
        -0.33794526671041136,
        0.41393543745829764,
        -0.45118386359032686,
        -0.33794526671041136,
        0.8238902915440387,
    ];
    let (_, raw) = crate::oracle::sym_eigen(&c);
    let lead = |v: &[f64]| {
        let mut l = 0;
        for (i, vi) in v.iter().enumerate() {
            if vi.abs() > v[l].abs() {
                l = i;
            }
        }
        v[l]
    };
    // The top component's: the eigenvalues come ascending.
    let top: Vec<f64> = raw[3].clone();
    assert!(
        lead(&top) < 0.0,
        "faer's own vector leads negative: {top:?}"
    );
    let zero = Pca {
        eig: vec![1.0],
        trace: 1.0,
        loadings: vec![0.0; 4],
    };
    let got = Pca::of(&c, 4, 1, Some(&zero)).unwrap();
    assert_eq!(got, Pca::of(&c, 4, 1, None).unwrap());
    assert!(lead(&got.loadings) > 0.0, "{got:?}");

    assert_eq!(Pca::of(&[1e308, 1e308, 1e308, 1e308], 2, 2, None), None);
}

/// The window's weight is the windowed accumulator's, an empty window's
/// 0 included: after rows of weight about `1e6`, rows of weight 0 for
/// longer than the window leave a crumb `±W·1e-16` where nothing is
/// inside it, which reads 0, not the crumb (`EMPTY_FRACTION` of the live
/// weight, not of its inverse).
#[test]
fn the_window_weight_is_the_windowed_accumulators_and_a_crumb_is_empty() {
    use crate::OnlineModel;
    let mut c = model_cfg(2, vec![EwCovStat::Mean]);
    c.decay = crate::Decay::Halflife(3.0);
    c.window = Some(5.0);
    let mut m = EwCovModel::new(c).unwrap();
    let mut s = 119u64;
    let (mut crumbs, mut empty) = (0, 0);
    for i in 0..400 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let w = if (i / 10) % 2 == 0 {
            1e6 * (1.5 + lcg(&mut s))
        } else {
            0.0
        };
        let want = m.windowed_cov().n_eff();
        assert_eq!(m.n_eff().to_bits(), want.to_bits(), "row {i}");
        if let Some((f, old)) = m.win.as_ref().and_then(|win| {
            let (f, old) = win.snaps.cut(win.clock, m.cfg.decay, 0)?;
            Some((f, old.w))
        }) {
            let live = m.cov.n_eff();
            let crumb = live - f * old;
            if crumb != 0.0 && crumb.abs() <= crate::window::EMPTY_FRACTION * live {
                crumbs += 1;
            }
        }
        empty += usize::from(want == 0.0);
        m.step(&x, &[], if i == 0 { 0.0 } else { 1.0 }, w);
    }
    assert!(
        crumbs > 0 && empty > 0,
        "{crumbs} crumbs, {empty} empty windows"
    );
}

/// A windowed model reports its ring past a refusing budget, the
/// snapshot spacing with it, and clears when the budget goes; its
/// shadow, learning the rows the model learns, is past it exactly when
/// the ring is (docs/PLAN.md task 115 (d)).
#[test]
fn an_ew_cov_window_budget_and_its_shadow() {
    use crate::OnlineModel;
    let plain = EwCovModel::new(model_cfg(2, vec![EwCovStat::Mean])).unwrap();
    assert!(plain.window_shadow().is_none());
    let mut c = model_cfg(2, vec![EwCovStat::Mean]);
    c.decay = crate::Decay::Halflife(30.0);
    c.window = Some(40.0);
    c.max_rows_between_snapshots = Some(2);
    let mut m = EwCovModel::new(c).unwrap();
    let mut s = 127u64;
    for i in 0..10 {
        m.step(
            &[lcg(&mut s), lcg(&mut s)],
            &[],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    assert_eq!(m.window_over_budget(), None, "no budget, no overrun");
    m.set_window_budget(Some(crate::WindowBudget::Refuse(1e-6)));
    match m.window_over_budget() {
        Some((bytes, every)) => {
            assert!(bytes > 1, "{bytes}");
            assert_eq!(every, crate::Cadence::of(None, Some(2)));
        }
        None => panic!("a ring of snapshots is past a budget of one byte"),
    }
    m.set_window_budget(Some(crate::WindowBudget::Refuse(0.0005)));
    let mut shadow = m.window_shadow().expect("a windowed model has a shadow");
    let mut over = 0;
    for _ in 0..30 {
        shadow.learn(1.0, None);
        m.step(&[lcg(&mut s), lcg(&mut s)], &[], 1.0, 1.0);
        assert_eq!(shadow.over_budget(), m.window_over_budget());
        over += usize::from(m.window_over_budget().is_some());
    }
    assert!(over > 0, "the ring went past its budget");
    m.set_window_budget(None);
    assert_eq!(m.window_over_budget(), None);
}

/// The partial correlations are one a pair, `k(k−1)/2` of them; the
/// components before their first refresh are `pca · (k + 3)` NaNs; and
/// each row is as wide as the cfg says, at `k = 3` (task 158).
#[test]
fn every_statistic_is_as_wide_as_the_cfg_says() {
    use crate::OnlineModel;
    let mut c = model_cfg(3, vec![EwCovStat::PartialCorr]);
    c.precision_prior = Some(1e-3);
    c.min_weight = 0.0;
    c.pca = 2;
    c.pca_every = 5.0;
    let mut m = EwCovModel::new(c).unwrap();
    let mut s = 131u64;
    for i in 0..20 {
        let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
        let out = m.step(&x, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0).pred;
        assert_eq!(out.len(), 3 + 2 * 6, "row {i}");
        if i == 0 {
            assert!(out[3..].iter().all(|v| v.is_nan()), "{out:?}");
        } else {
            assert!(out[..3].iter().all(|v| (-1.0..=1.0).contains(v)), "{out:?}");
        }
    }
}

/// The partial correlations fill their slots pair by pair, `(0,1), (0,2),
/// …, (k−2,k−1)`, each the pair's value from the precision the row is
/// predicted with (the moments before its update): a slot shifted by one
/// pair, or a diagonal read as a pair, reads another pair's number.
#[test]
fn the_partial_correlations_fill_their_slots_in_pair_order() {
    use crate::OnlineModel;
    let k = 4;
    let mut c = model_cfg(k, vec![EwCovStat::PartialCorr]);
    c.precision_prior = Some(1e-3);
    c.min_weight = 0.0;
    let mut m = EwCovModel::new(c).unwrap();
    let mut s = 977u64;
    let mut checked = 0;
    for row in 0..40 {
        let x: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
        let before = m.cov.precision();
        let out = m.step(&x, &[], 1.0, 1.0).pred;
        assert_eq!(out.len(), k * (k - 1) / 2, "row {row}");
        let Some(p) = before else { continue };
        let want: Vec<f64> = (0..k)
            .flat_map(|i| (i + 1..k).map(move |j| (i, j)))
            .map(|(i, j)| partial_corr(&p, k, i, j))
            .collect();
        assert!(want.iter().all(|v| v.is_finite()), "row {row}: {want:?}");
        for (slot, (got, want)) in out.iter().zip(&want).enumerate() {
            assert_eq!(got.to_bits(), want.to_bits(), "row {row}, slot {slot}");
        }
        checked += 1;
    }
    assert!(checked >= 30, "only {checked} rows had a precision");
}

/// A column whose variance underflows to 0 has no lagged correlation,
/// even where its lagged co-moment with a wide partner does not
/// underflow: deviations near `1e-170` square to 0, while their product
/// with deviations near `1e90` is `1e-80` (task 158).
#[test]
fn a_column_without_variance_has_no_lagged_correlation() {
    use crate::OnlineModel;
    let mut c = model_cfg(2, vec![EwCovStat::LagCorr]);
    c.lags = vec![1];
    c.min_weight = 0.0;
    c.decay = crate::Decay::Halflife(20.0);
    let mut m = EwCovModel::new(c).unwrap();
    let mut s = 137u64;
    let mut out = Vec::new();
    for i in 0..30 {
        let u = lcg(&mut s);
        let x = [1e-170 * u, 1e90 * (u + 0.1 * lcg(&mut s))];
        out = m.step(&x, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0).pred;
    }
    assert_eq!(
        m.cov.var(0),
        0.0,
        "the narrow column's variance underflowed"
    );
    assert!(m.cov.var(1) > 0.0);
    let lag = m.lag.as_ref().unwrap();
    assert!(
        lag.get(0, 0, 1) != 0.0 || lag.get(0, 1, 0) != 0.0,
        "a co-moment survived"
    );
    // Slots `a·k + b`: (0, 0), (0, 1), (1, 0), (1, 1).
    assert!(
        out[0].is_nan() && out[1].is_nan() && out[2].is_nan(),
        "{out:?}"
    );
    assert!(out[3].is_finite(), "{out:?}");
}

/// A model without components keeps no component counter in its state
/// (task 158), and takes no stamp for one (task 180): a stamp left
/// waiting would part the model from its own restored copy, which holds
/// none.
#[test]
fn a_model_without_components_counts_no_refreshes() {
    use crate::OnlineModel;
    let mut c = model_cfg(2, vec![EwCovStat::Mean]);
    c.min_weight = 10.0;
    let mut m = EwCovModel::new(c).unwrap();
    for i in 0..5 {
        m.stamp_next(crate::Stamp::Ns(i128::from(i) * 1_000_000_000));
        m.step(&[1.0, f64::from(i)], &[], 1.0, 1.0);
    }
    assert_eq!(m.rows_since_pca, 0);
    assert_eq!(m.since_pca, Since::default());
}

/// A state with components loads when they are its cfg's (here `k = 3`
/// and two components, where `r · k` is not `r + k`), and is refused
/// when its loadings are short, its lagged moments are another shape
/// than its lags, or it has a ring its cfg has no window for -- each on
/// its own (task 158).
#[test]
fn a_state_is_refused_for_each_part_of_the_wrong_shape() {
    use crate::{ModelState, OnlineModel, StateError};
    let mut c = pca_cfg(3, 2, 1);
    c.min_weight = 0.0;
    c.lags = vec![1];
    let mut m = EwCovModel::new(c).unwrap();
    let mut s = 139u64;
    for i in 0..10 {
        m.step(
            &[lcg(&mut s), lcg(&mut s), lcg(&mut s)],
            &[],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    assert_eq!(m.pca.as_ref().map(Pca::r), Some(2));
    let back = EwCovModel::restore(&m.state()).expect("the state's own shape loads");
    assert_eq!(back.pca, m.pca);
    let refused = |edit: &dyn Fn(&mut EwCovModel)| {
        let mut st = m.state();
        let ModelState::EwCovModel(inner) = &mut st.model else {
            unreachable!()
        };
        edit(inner);
        match EwCovModel::restore(&st) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
            other => panic!("{other:?}"),
        }
    };
    refused(&|inner| {
        inner.pca.as_mut().unwrap().loadings.pop();
    });
    refused(&|inner| inner.lag = Some(crate::EwLagCov::new(3, vec![2]).unwrap()));
    refused(&|inner| {
        let mut w = EwCovModel::new({
            let mut c = model_cfg(3, vec![EwCovStat::Mean]);
            c.window = Some(10.0);
            c
        })
        .unwrap();
        inner.win = w.win.take();
    });
}

/// Every problem at once: two bad settings are both named, one to a line.
#[test]
fn a_configuration_with_two_problems_reports_both() {
    let mut c = model_cfg(3, vec![EwCovStat::Mean]);
    c.min_weight = -1.0;
    c.pca = 4;
    let e = c.validate().unwrap_err();
    let lines: Vec<&str> = e.lines().collect();
    assert_eq!(lines.len(), 2, "{e}");
    assert!(
        lines[0].starts_with("ew_cov: min_weight must be at least 0"),
        "{e}"
    );
    assert!(lines[1].starts_with("ew_cov: pca must be at most"), "{e}");
}

/// A windowed model's view, kept between rows and written over
/// (`crate::window::KeptView`), reads every row's statistics as a view made
/// afresh does, to the bit: every statistic, through windows that rows of
/// weight 0 after a gap leave empty, and a held feature (task 374).
#[test]
fn the_kept_window_reads_as_a_fresh_one() {
    use crate::OnlineModel;
    let stats = vec![
        EwCovStat::Mean,
        EwCovStat::Var,
        EwCovStat::Std,
        EwCovStat::Cov,
        EwCovStat::Corr,
        EwCovStat::PartialCorr,
        EwCovStat::Mahal,
    ];
    let mut cfg = model_cfg(5, stats);
    cfg.decay = crate::Decay::Halflife(40.0);
    cfg.min_weight = 0.0;
    cfg.precision_prior = Some(0.25);
    cfg.window = Some(12.0);
    let mut kept = EwCovModel::new(cfg).unwrap();
    let mut fresh = kept.clone();
    let mut s = 5u64;
    let mut empty = 0;
    for i in 0..400 {
        fresh.view = crate::window::Kept::default();
        let x = [
            lcg(&mut s),
            3.0,
            1e6 + lcg(&mut s),
            lcg(&mut s) - 0.5 * lcg(&mut s),
            2.0,
        ];
        let (d, w) = match i % 100 {
            90 => (50.0, 0.0),
            91..=94 => (1.0, 0.0),
            _ => (1.0, 1.0 + 0.5 * lcg(&mut s)),
        };
        let a = kept.step(&x, &[], d, w);
        let b = fresh.step(&x, &[], d, w);
        let bits = |v: &[f64]| v.iter().map(|p| p.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.pred), bits(&b.pred), "row {i}");
        assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits(), "row {i}");
        empty += usize::from(a.n_eff == 0.0 && i > 50);
    }
    assert!(empty > 5, "{empty}");
    assert_eq!(kept, fresh);
}
