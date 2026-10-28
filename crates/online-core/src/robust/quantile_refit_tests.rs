//! `quantile`'s refit (`robust::quantile_refit`; docs/PLAN.md task 278,
//! D-Q2).

use super::*;
use crate::OnlineModel;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn base(tau: f64, decay: Decay, refit_rows: u32) -> RobustCfg {
    RobustCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay,
        loss: RobustLoss::Quantile { tau },
        ridge: 1e-6,
        standardize: false,
        min_weight: 0.0,
        solve_every: 0.0,
        max_rows_between_solves: u32::MAX,
        solve_share: None,
        quantile_eps: 0.2,
        huber_scale: HuberScale::Mad,
        huber_scale_decay: None,
        feature_sets: Vec::new(),
        ridge_scale: false,
        emit_scale: false,
        refit_rows,
        mad_anchor: true,
        min_weight_holds_learning: false,
    }
}

/// Laplace-ish noise: a difference of two exponentials, from the LCG.
fn laplace(s: &mut u64) -> f64 {
    let u = 0.5 * (lcg(s) + 1.0);
    let v = 0.5 * (lcg(s) + 1.0);
    0.5 * ((u.max(1e-300)).ln() - (v.max(1e-300)).ln())
}

/// One row: the features, the targets and the weight.
type Row = ([f64; 2], Vec<Option<f64>>, f64);

/// `y = 1 + 2x₀ − x₁ + 0.5·Laplace` on one target, or two (the second
/// `−3 + x₁ + 0.5·Laplace`, null every 17 rows), the first `wild` rows'
/// first target `1e4` off, and a zero-weight row every 9 where asked.
fn stream(n: usize, seed: u64, wild: usize, targets: usize, zeros: bool) -> Vec<Row> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            let x = [lcg(&mut s), lcg(&mut s)];
            let off = if i < wild { 1e4 } else { 0.0 };
            let y0 = 1.0 + 2.0 * x[0] - x[1] + 0.5 * laplace(&mut s) + off;
            let mut y = vec![Some(y0)];
            if targets == 2 {
                let y1 = -3.0 + x[1] + 0.5 * laplace(&mut s);
                y.push((i % 17 != 3).then_some(y1));
            }
            let w = if zeros && i % 9 == 4 { 0.0 } else { 1.0 };
            (x, y, w)
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

fn kinds() -> [u32; 4] {
    super::quantile_refit::PASS_KINDS.with(std::cell::Cell::get)
}

fn reset_kinds() {
    super::quantile_refit::PASS_KINDS.with(|c| c.set([0; 4]));
}

/// The refit fires on the row a target's learned rows reach the count, a
/// row outside the band counted as one: the predictions part from the run
/// without it on the next row and not before. It moves its own target
/// alone -- the other's predictions are that run's to the bit until its own
/// buffer fills -- and the solve cadence is that run's on every row.
#[test]
fn the_refit_fires_on_its_count_and_moves_its_own_target_alone() {
    let data = stream(400, 3, 0, 2, false);
    let mut c = base(0.5, Decay::Lam(0.999), 0);
    c.n_targets = 2;
    c.max_rows_between_solves = 7;
    c.solve_every = f64::INFINITY;
    let mut off = Robust::new(c.clone()).unwrap();
    c.refit_rows = 100;
    let mut on = Robust::new(c).unwrap();
    let mut fills = [None, None];
    let mut first_move = [None, None];
    for (i, (x, y, w)) in data.iter().enumerate() {
        let (a, b) = (on.step(x, y, 1.0, *w).pred, off.step(x, y, 1.0, *w).pred);
        assert_eq!(on.rows_since_solve, off.rows_since_solve, "row {i}");
        assert_eq!(on.since_solve, off.since_solve, "row {i}");
        for j in 0..2 {
            if first_move[j].is_none() && a[j].to_bits() != b[j].to_bits() {
                first_move[j] = Some(i);
            }
            if fills[j].is_none() && !on.refit[j].open() {
                fills[j] = Some(i);
            }
        }
    }
    // Target 0 is on every row and fills on row 99; target 1 is null on
    // rows 3, 20, 37, 54, 71, 88 and 105, so its 100th row is row 106.
    assert_eq!(fills, [Some(99), Some(106)]);
    assert_eq!(first_move, [Some(100), Some(107)]);
    let st = on.refit_status();
    assert!(
        st.iter().all(|s| s.done && s.passes > 0 && !s.failed),
        "{st:?}"
    );
}

/// The refit's fit is the minimum of the smoothed check loss over its rows
/// at the band its own residuals draw (the module docs of
/// `quantile_refit`): one Newton step of that loss from the fit, solved by
/// `crate::oracle`'s LU on the raw system written from the definition,
/// returns the fit, at two levels, with and without decay.
#[test]
fn the_refit_is_the_smoothed_check_losss_stationary_point() {
    for (tau, lam) in [(0.5, 1.0), (0.8, 1.0), (0.5, 0.995), (0.8, 0.995)] {
        let n = 300;
        let data = stream(n, 11, 5, 1, false);
        let mut c = base(tau, Decay::Lam(lam), n as u32);
        c.ridge = 0.0;
        let mut m = Robust::new(c).unwrap();
        run(&mut m, &data);
        let st = m.refit_status()[0];
        assert!(st.done && st.passes > 0 && !st.ran_to_cap(), "{st:?}");
        let fit = m.coefficients().unwrap()[0].clone();
        // The rows' weights decayed to the last, and counted one each.
        let v: Vec<f64> = (0..n).map(|i| lam.powi((n - 1 - i) as i32)).collect();
        let count: f64 = v.iter().sum();
        let z = |i: usize| [1.0, data[i].0[0], data[i].0[1]];
        let y = |i: usize| data[i].1[0].unwrap();
        let r: Vec<f64> = (0..n)
            .map(|i| y(i) - z(i).iter().zip(&fit).map(|(a, b)| a * b).sum::<f64>())
            .collect();
        let absr: Vec<f64> = r.iter().map(|e| e.abs()).collect();
        let s = crate::huber_refit::weighted_median(&absr, &v) / NORMAL_Q3;
        let h = s * 0.2f64.max((3.0 / count).powf(0.4));
        let (mut a, mut b) = (vec![0.0; 9], vec![0.0; 3]);
        for i in 0..n {
            let zi = z(i);
            if r[i].abs() < h {
                for p in 0..3 {
                    for q in 0..3 {
                        a[p * 3 + q] += v[i] * zi[p] * zi[q];
                    }
                    b[p] += v[i] * zi[p] * (y(i) + 2.0 * h * (tau - 0.5));
                }
            } else {
                let psi = if r[i] > 0.0 { tau } else { tau - 1.0 };
                for p in 0..3 {
                    b[p] += v[i] * zi[p] * 2.0 * h * psi;
                }
            }
        }
        let step = crate::oracle::solve(&a, &b);
        let size = fit.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let moved = step
            .iter()
            .zip(&fit)
            .fold(0.0f64, |m, (p, q)| m.max((p - q).abs()));
        assert!(
            moved <= 1e-9 * size,
            "tau {tau}, lam {lam}: {fit:?} -> {step:?}"
        );
    }
}

/// Five targets 1e4 off in the first rows: the stream takes them as least
/// squares in its warm-up and keeps them in its sums, and without decay
/// they leave only by dilution; the refit scores them afresh as rows
/// outside the band, and the fit is the clean stream's to within its noise.
#[test]
fn wild_first_rows_leave_with_the_refit() {
    let n = 3000;
    let gap = |refit: u32, wild: usize| {
        let mut m = Robust::new(base(0.5, Decay::Lam(1.0), refit)).unwrap();
        run(&mut m, &stream(n, 5, wild, 1, false));
        m.coefficients().unwrap()[0].clone()
    };
    let clean = gap(500, 0);
    let dist = |a: &[f64]| {
        a.iter()
            .zip(&clean)
            .fold(0.0f64, |m, (p, q)| m.max((p - q).abs()))
    };
    let (off, on) = (dist(&gap(0, 5)), dist(&gap(500, 5)));
    assert!(off > 0.1, "the wild rows held the one pass: {off}");
    assert!(on < 0.01, "the refit let them go: {on} (without it {off})");
}

/// From a fit far from its rows, a full Newton step overshoots -- the band's
/// rows are a curvature measured at the fit, not at the step -- and the
/// refit halves it; from a fit whose residuals all sit on one side, the band
/// holds no row and the refit takes the majorizer's step. Either way it
/// reaches the fit it reaches from the stream's own, to the tolerance.
#[test]
fn a_far_start_is_damped_and_an_empty_band_majorized() {
    let n = 400;
    let data = stream(n, 9, 0, 1, false);
    let mut c = base(0.5, Decay::Lam(1.0), n as u32);
    // No solve after the first rows': the refit starts from the fit as the
    // test leaves it.
    c.solve_every = f64::INFINITY;
    let refit_from = |shift: [f64; 3]| {
        let mut m = Robust::new(c.clone()).unwrap();
        run(&mut m, &data[..n - 1]);
        if let Some(b) = m.beta.as_mut() {
            for (v, d) in b[0].iter_mut().zip(shift) {
                *v += d;
            }
        }
        reset_kinds();
        run(&mut m, &data[n - 1..]);
        let st = m.refit_status()[0];
        assert!(
            st.done && !st.failed && !st.ran_to_cap(),
            "{shift:?}: {st:?}"
        );
        (m.coefficients().unwrap()[0].clone(), kinds())
    };
    let (plain, k0) = refit_from([0.0; 3]);
    assert_eq!(k0[2], 0, "the stream's own fit has a band");
    let (far, kf) = refit_from([0.0, 30.0, -30.0]);
    assert!(kf[3] > 0, "a step was halved: {kf:?}");
    let (level, kl) = refit_from([100.0, 0.0, 0.0]);
    assert!(kl[2] > 0, "the majorizer stepped: {kl:?}");
    for (name, b) in [("far", &far), ("level", &level)] {
        let d = b
            .iter()
            .zip(&plain)
            .fold(0.0f64, |m, (p, q)| m.max((p - q).abs()));
        assert!(d <= 1e-8, "{name}: {b:?} against {plain:?}");
    }
}

/// Saved and loaded mid-buffer, the model goes on as one that never
/// stopped, to the bit.
#[test]
fn a_save_and_load_mid_buffer_goes_on_to_the_bit() {
    let data = stream(300, 4, 3, 2, true);
    let mut cfg = base(0.7, Decay::Lam(0.999), 120);
    cfg.n_targets = 2;
    let whole = run(&mut Robust::new(cfg.clone()).unwrap(), &data);
    let mut m = Robust::new(cfg).unwrap();
    let mut parts = run(&mut m, &data[..70]);
    assert!(m.refit.iter().all(|b| b.open() && b.held() > 0));
    let bytes = rmp_serde::to_vec(&m.state()).unwrap();
    let mut back = Robust::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    parts.extend(run(&mut back, &data[70..]));
    assert_eq!(bits(&whole), bits(&parts));
    assert!(back.refit_status().iter().all(|s| s.done && s.passes > 0));
}

/// A zero-weight row is clock alone (hard rule 9), inside the band or
/// outside it: not buffered and not counted, so without decay a stream with
/// such rows -- a first one among them -- predicts every other row as the
/// stream without them does, to the bit, the refit included.
#[test]
fn zero_weight_rows_neither_count_nor_move_the_refit() {
    let with = stream(300, 6, 3, 2, true);
    let without: Vec<_> = with.iter().filter(|r| r.2 > 0.0).cloned().collect();
    let mut cfg = base(0.5, Decay::Lam(1.0), 80);
    cfg.n_targets = 2;
    let mut lead = vec![([0.3, -0.2], vec![Some(1000.0), Some(0.0)], 0.0)];
    lead.extend(with.iter().cloned());
    let mut a_model = Robust::new(cfg.clone()).unwrap();
    let a = run(&mut a_model, &lead);
    let b = run(&mut Robust::new(cfg).unwrap(), &without);
    let kept: Vec<Vec<f64>> = lead
        .iter()
        .zip(a)
        .filter(|(r, _)| r.2 > 0.0)
        .map(|(_, p)| p)
        .collect();
    assert_eq!(bits(&kept), bits(&b));
    assert!(
        a_model
            .refit_status()
            .iter()
            .all(|s| s.done && s.passes > 0)
    );
}

/// A buffer the stream still takes as its warm-up's least squares -- its
/// rows counted one each and decayed short of three a coefficient, under a
/// half-life of two rows -- is left as the stream built it: the run is the
/// one without the refit, to the bit, its buffer let go without a pass.
#[test]
fn a_buffer_inside_the_warm_up_is_left_alone() {
    let data = stream(200, 8, 0, 1, false);
    let off = run(
        &mut Robust::new(base(0.5, Decay::Halflife(2.0), 0)).unwrap(),
        &data,
    );
    let mut m = Robust::new(base(0.5, Decay::Halflife(2.0), 20)).unwrap();
    let on = run(&mut m, &data);
    assert_eq!(bits(&off), bits(&on));
    let st = m.refit_status()[0];
    assert!(st.done && st.passes == 0 && !st.failed, "{st:?}");
}

/// The count is at least the warm-up's three rows a coefficient: nine here,
/// two features and an intercept.
#[test]
fn a_count_inside_the_warm_up_is_refused() {
    assert!(Robust::new(base(0.5, Decay::Lam(1.0), 8)).is_err());
    assert!(Robust::new(base(0.5, Decay::Lam(1.0), 9)).is_ok());
    assert!(Robust::new(base(0.5, Decay::Lam(1.0), 0)).is_ok());
    let mut origin = base(0.5, Decay::Lam(1.0), 6);
    origin.fit_intercept = false;
    assert!(Robust::new(origin.clone()).is_ok());
    origin.refit_rows = 5;
    assert!(Robust::new(origin).is_err());
}

/// Over 3,000 short streams with a level added to up to 45% of the rows on
/// one side (`slow_refit_stream`, which takes `huber`'s IRLS up to 428
/// passes), no refit runs to its cap: the most passes taken were 19. Nor
/// on the first 1,000 of them moved to a level of `1e6`, where a step's
/// rounding can stay above the stop pass after pass and the refit ends on
/// a step that no longer lowers its loss (task 278's follow-ups): 15 at
/// most, where without that end 368 of the 1,000 ran to the cap.
#[test]
fn the_refit_converges_on_hard_short_streams() {
    let mut most = 0;
    let streams = (0..3000)
        .map(|t| (t, 0.0))
        .chain((0..1000).map(|t| (t, 1e6)));
    for (trial, level) in streams {
        let rows = crate::huber_refit::slow_refit_stream(trial);
        let mut c = base(0.5, Decay::Lam(1.0), rows.len() as u32);
        c.n_features = 1;
        let mut m = Robust::new(c).unwrap();
        for (x, y) in &rows {
            m.step(&[*x], &[Some(*y + level)], 1.0, 1.0);
        }
        let st = m.refit_status()[0];
        assert!(
            st.done && !st.failed && !st.ran_to_cap(),
            "trial {trial} at {level}: {st:?}"
        );
        assert_eq!(m.solve_failures, 0, "trial {trial} at {level}");
        most = most.max(st.passes);
    }
    assert!(most <= 40, "{most}");
}

/// The refit's stop reads no level (task 278's follow-ups): the same rows
/// moved to `±1e3` refit to the fit at 0 moved by the level, the slopes to
/// `1e-13` and the intercept to its last places, in the same passes. The
/// stop relative to `max|β|` left the slopes `1.2e-11` apart at `1e3`, and
/// the intercept `2.0e-11`.
#[test]
fn a_level_does_not_loosen_the_refit() {
    let n = 400;
    let data = stream(n, 7, 3, 1, false);
    let fit = |level: f64| {
        let shifted: Vec<Row> = data
            .iter()
            .map(|(x, y, w)| (*x, vec![y[0].map(|v| v + level)], *w))
            .collect();
        let mut m = Robust::new(base(0.75, Decay::Lam(1.0), n as u32)).unwrap();
        run(&mut m, &shifted);
        let st = m.refit_status()[0];
        assert!(st.done && !st.ran_to_cap(), "{level}: {st:?}");
        (m.coefficients().unwrap()[0].clone(), st.passes)
    };
    let (at0, p0) = fit(0.0);
    for level in [1e3, -1e3] {
        let (b, p) = fit(level);
        assert!(
            (b[0] - level - at0[0]).abs() <= 1e-11,
            "{level}: {b:?} against {at0:?}"
        );
        for (q, r) in b[1..].iter().zip(&at0[1..]) {
            assert!((q - r).abs() <= 1e-13, "{level}: {b:?} against {at0:?}");
        }
        assert_eq!(p, p0, "{level}");
    }
}

/// A level far above the noise moves the refit's fit by the level and by
/// rounding alone: the intercept to a few hundred units in the last place
/// of the level, the slopes likewise. The test's bound is `128 u (|L| +
/// max|y|)`, the bound the convergence classes hold the stream to. Its
/// "no descent" stop compared the loss at two fits, each a sum of
/// residuals `y - z'β` rounded at the level: at `1e6` that rounding was
/// `1e-10` a row, ten thousand times the loss's own slack, so a Newton step
/// of `1e-5` that did lower the loss read as one that did not, and the
/// refit stopped there; the stream then carried that fit on (task 278,
/// qequiv: up to `4.5e7` of the bound in the convergence class).
#[test]
fn a_far_level_moves_the_refit_by_rounding_alone() {
    let n = 500;
    let (mut worst, mut over) = (0.0f64, Vec::new());
    for seed in 0..20u64 {
        for tau in [0.5, 0.9] {
            let data = stream(n, 100 + seed, 0, 1, false);
            let top = data
                .iter()
                .fold(0.0f64, |m, (_, y, _)| m.max(y[0].unwrap().abs()));
            let fit = |level: f64| {
                let shifted: Vec<Row> = data
                    .iter()
                    .map(|(x, y, w)| (*x, vec![y[0].map(|v| v + level)], *w))
                    .collect();
                let mut m = Robust::new(base(tau, Decay::Lam(1.0), n as u32)).unwrap();
                run(&mut m, &shifted);
                let st = m.refit_status()[0];
                assert!(st.done && !st.ran_to_cap(), "{level}: {st:?}");
                m.coefficients().unwrap()[0].clone()
            };
            let at0 = fit(0.0);
            for level in [1e6, -1e9] {
                let b = fit(level);
                let unit = f64::EPSILON / 2.0 * (level.abs() + top);
                let off = (b[0] - level - at0[0])
                    .abs()
                    .max((b[1] - at0[1]).abs())
                    .max((b[2] - at0[2]).abs())
                    / unit;
                worst = worst.max(off);
                if off > 128.0 {
                    over.push(format!("seed {seed}, tau {tau}, {level}: {off:.0} units"));
                }
            }
        }
    }
    println!("worst: {worst} units");
    assert!(over.is_empty(), "{over:#?}");
}

/// The warm-up's bound is three rows a coefficient, counted one each and
/// decayed: at a count of nine rows without decay -- exactly the bound, two
/// features and an intercept -- the refit runs; at twenty rows decayed by
/// 0.88 a row, `(1 − 0.88²⁰)/0.12 = 7.69` of them, under the bound but past
/// three rows plus a coefficient each, it leaves the buffer as the stream
/// built it (task 281, mutants).
#[test]
fn the_warm_ups_bound_is_three_rows_a_coefficient() {
    let at = |decay: Decay, rows: u32| {
        let mut m = Robust::new(base(0.5, decay, rows)).unwrap();
        run(&mut m, &stream(rows as usize + 5, 12, 0, 1, false));
        m.refit_status()[0]
    };
    let nine = at(Decay::Lam(1.0), 9);
    assert!(nine.done && nine.passes > 0, "{nine:?}");
    let decayed = at(Decay::Lam(0.88), 20);
    assert!(decayed.done && decayed.passes == 0, "{decayed:?}");
}

/// The band's rows are counted one each, not by weight: a stream at weight
/// 0.1 a row takes Newton steps, as the stream at weight 1 does, where its
/// weights summed would leave the band short of a row a coefficient and
/// take the majorizer's (task 281, mutants).
#[test]
fn the_bands_rows_are_counted_one_each() {
    let n = 300;
    let kinds_at = |w: f64| {
        let data: Vec<Row> = stream(n, 13, 0, 1, false)
            .into_iter()
            .map(|(x, y, _)| (x, y, w))
            .collect();
        reset_kinds();
        let mut m = Robust::new(base(0.5, Decay::Lam(1.0), n as u32)).unwrap();
        run(&mut m, &data);
        assert!(m.refit_status()[0].done);
        kinds()
    };
    for w in [1.0, 0.1] {
        let k = kinds_at(w);
        assert!(k[1] > 0 && k[2] == 0, "{w}: {k:?}");
    }
}

/// A band holding exactly a row a coefficient has the curvature for a
/// Newton step, and takes it: from a fit of 0 on nine rows whose residuals'
/// sizes are 0.1, 0.2, 0.3, 0.96, 1, 1.1, 1.2, 1.3 and 1.4, the band is
/// `0.6444 · 1/Φ⁻¹(3/4) = 0.9554` wide and holds three rows (task 281,
/// mutants).
#[test]
fn a_band_of_a_row_a_coefficient_takes_the_newton_step() {
    let r = [0.1, -0.2, 0.3, -0.96, 1.0, -1.1, 1.2, -1.3, 1.4];
    let x = [
        [0.5, -0.1],
        [-0.4, 0.7],
        [0.2, 0.3],
        [0.9, -0.8],
        [-0.6, -0.5],
        [0.1, 0.9],
        [-0.9, 0.2],
        [0.7, 0.6],
        [-0.2, -0.9],
    ];
    let mut c = base(0.5, Decay::Lam(1.0), 9);
    c.solve_every = f64::INFINITY;
    let mut m = Robust::new(c).unwrap();
    for i in 0..8 {
        m.step(&x[i], &[Some(r[i])], 1.0, 1.0);
    }
    if let Some(b) = m.beta.as_mut() {
        b[0].fill(0.0);
    }
    reset_kinds();
    m.step(&x[8], &[Some(r[8])], 1.0, 1.0);
    let st = m.refit_status()[0];
    assert!(st.done && st.passes > 0, "{st:?}");
    assert_eq!(kinds()[2], 0, "{:?}", kinds());
    assert!(kinds()[1] > 0, "{:?}", kinds());
}

/// A fit no row of which lies inside its band -- targets of +1 and −1 in
/// pairs of rows with the same features, so the smoothed loss is flat
/// wherever every fitted value is within `1 − h` of 0, and the band, `0.37`
/// wide here, holds no row there -- is rebuilt as the majorizer's rows,
/// as a pass with such a band is: the target keeps its rows' weight and
/// predicts the next row, as the stream without the refit does. Rebuilt as
/// the stream takes each row, every row a nudge, the sums had no weight and
/// the next row no prediction (task 281).
#[test]
fn a_band_with_no_row_keeps_the_targets_weight() {
    let n = 100;
    let data: Vec<Row> = (0..n)
        .map(|i| {
            let k = i / 2;
            let x = [
                f64::from(k % 10) / 10.0 - 0.45,
                f64::from(k % 7) / 7.0 - 0.43,
            ];
            (x, vec![Some(if i % 2 == 0 { 1.0 } else { -1.0 })], 1.0)
        })
        .collect();
    let mut on = Robust::new(base(0.5, Decay::Lam(1.0), n)).unwrap();
    let mut off = Robust::new(base(0.5, Decay::Lam(1.0), 0)).unwrap();
    run(&mut on, &data);
    run(&mut off, &data);
    let st = on.refit_status()[0];
    assert!(st.done && st.passes > 0 && !st.failed, "{st:?}");
    // The premise: the converged fit's band holds no row.
    let b = on.coefficients().unwrap()[0].clone();
    let fitted = |x: &[f64; 2]| b[0] + b[1] * x[0] + b[2] * x[1];
    assert!(
        data.iter()
            .all(|(x, y, _)| (y[0].unwrap() - fitted(x)).abs() > 0.5)
    );
    assert!(
        on.wj[0] > 0.0 && off.wj[0] > 0.0,
        "{} {}",
        on.wj[0],
        off.wj[0]
    );
    assert!(on.ybar[0].is_finite() && on.cross[0].iter().all(|c| c.is_finite()));
    let x = [0.1, 0.1];
    let (p_on, p_off) = (on.predict(&x, 1.0).pred, off.predict(&x, 1.0).pred);
    assert!(
        p_on[0].is_finite() && p_off[0].is_finite(),
        "{p_on:?} {p_off:?}"
    );
    assert!(p_on[0].abs() < 1.0, "a median of ±1 pairs: {p_on:?}");
}

/// A refit that takes every pass is counted once in `solve_failures`, on
/// its row, over what the model without the refit counts there, and its
/// last pass's fit stands: 16 rows with four targets 30 off, `τ = 0.75` and
/// `quantile_eps = 1e-3`, found by a search over short streams (task 281,
/// mutants), which do not settle in the 200 passes.
#[test]
fn a_refit_that_runs_to_its_cap_is_counted_once() {
    let rows = [
        (0.861098701750727, 3.1475333096309686),
        (0.6664494252134896, 2.031214985879759),
        (0.9524682482377222, 0.7155630121703367),
        (-0.09280079788583251, 0.802322542552432),
        (0.7200361619605207, -0.056208739995097456),
        (-0.7924813270775997, 29.657911970641145),
        (-0.8891777654448711, -0.9511709878836129),
        (0.3360907396076267, 32.140838396357466),
        (-0.7808990340251272, -0.7056600423896854),
        (0.32538334264844093, -1.5390300862280992),
        (0.2670157442329868, 1.9396440664778627),
        (0.38218516486505805, 1.7507086468196766),
        (-0.6225324232431733, -0.6389818620568497),
        (0.3665882440091026, 31.0243119047125),
        (0.7391180132036732, 32.08472075046633),
        (0.22566527521606128, 1.1855941179582279),
    ];
    let mut c = base(0.75, Decay::Lam(1.0), 0);
    c.quantile_eps = 1e-3;
    c.n_features = 1;
    let mut off = Robust::new(c.clone()).unwrap();
    c.refit_rows = rows.len() as u32;
    let mut on = Robust::new(c).unwrap();
    for (i, (x, y)) in rows.iter().enumerate() {
        on.step(&[*x], &[Some(*y)], 1.0, 1.0);
        off.step(&[*x], &[Some(*y)], 1.0, 1.0);
        if i + 1 < rows.len() {
            assert_eq!(on.solve_failures, off.solve_failures, "row {i}");
        }
    }
    let st = on.refit_status()[0];
    assert!(st.done && st.ran_to_cap() && !st.failed, "{st:?}");
    assert_eq!(on.solve_failures, off.solve_failures + 1);
}
