//! `sgd`'s Huber refit (`SgdCfg::refit_rows`, `crate::huber_refit`;
//! docs/PLAN.md task 277): the refit sets the coefficients to the buffer's
//! converged fit whatever coordinates the scaler holds them in, leaves the
//! learning rate's state alone, fires on a count of learned rows, holds
//! through a save and a load, ignores zero-weight rows, and projects on a
//! constraint.

use super::*;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn cfg(refit_rows: u32, standardize: bool, schedule: LearningRate, decay: Decay) -> SgdCfg {
    SgdCfg {
        n_features: 2,
        n_targets: 2,
        fit_intercept: true,
        decay,
        loss: SgdLoss::Huber { delta: 1.345 },
        learning_rate: 0.05,
        schedule,
        l2: 0.0,
        min_weight: 3.0,
        clip_gradient: 1e3,
        constraint: None,
        standardize,
        strict_binary: false,
        huber_scale: crate::HuberScale::Mad,
        huber_scale_decay: None,
        refit_rows,
        min_weight_holds_learning: false,
        target_spread: false,
    }
}

/// One row: the features, the two targets and the weight.
type Row = ([f64; 2], [Option<f64>; 2], f64);

/// Two targets at a level of 1,000 and -3 with one-sided outliers on the
/// first, a null target every 17 rows and, where asked, a zero weight every 9.
fn rows(n: usize, zeros: bool) -> Vec<Row> {
    let mut s = 11u64;
    (0..n)
        .map(|i| {
            let x = [lcg(&mut s), 3.0 + lcg(&mut s)];
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

fn run(m: &mut Sgd, data: &[Row]) -> Vec<Vec<f64>> {
    data.iter()
        .map(|(x, y, w)| m.step(x, y, 1.0, *w).pred)
        .collect()
}

fn bits(p: &[Vec<f64>]) -> Vec<Vec<u64>> {
    p.iter()
        .map(|r| r.iter().map(|v| v.to_bits()).collect())
        .collect()
}

/// The refit's fit is the buffer's, whatever the coordinates: refit on the
/// tenth learned row, while the scaler still warms up (its count reaches 22
/// later), the standardized model's coefficients read out as the plain
/// model's, both being the converged fit of the same rows; and the one
/// refit after the switch, at 40, likewise.
#[test]
fn the_refit_reads_out_as_the_fit_whatever_the_scaler_holds() {
    for at in [10, 40] {
        let data = rows(60, false);
        let mut a = Sgd::new(cfg(at, true, LearningRate::Constant, Decay::Lam(1.0))).unwrap();
        let mut b = Sgd::new(cfg(at, false, LearningRate::Constant, Decay::Lam(1.0))).unwrap();
        // Target 1 is present on every row: its buffer fills on row at - 1.
        for (x, y, w) in &data[..at as usize] {
            a.step(x, y, 1.0, *w);
            b.step(x, y, 1.0, *w);
        }
        assert_eq!(a.scaler_warming(), at == 10, "the refit at {at}");
        let (ca, cb) = (a.coefficients(), b.coefficients());
        for (p, q) in ca[1].iter().zip(&cb[1]) {
            assert!(
                (p - q).abs() < 1e-9 * (1.0 + q.abs()),
                "{at}: {ca:?} {cb:?}"
            );
        }
        assert!(!a.refit[1].open() && !b.refit[1].open());
    }
}

/// The learning rate's state is not the refit's: on the row a buffer fills,
/// AdaGrad's sums are the run without a refit's to the bit, and the weight
/// `inv_scaling` reads is too (hard rule 8).
#[test]
fn the_refit_leaves_the_learning_rates_state() {
    for schedule in [
        LearningRate::AdaGrad,
        LearningRate::InvScaling { power: 0.5 },
    ] {
        let data = rows(60, true);
        let mut on = Sgd::new(cfg(30, false, schedule, Decay::Lam(0.99))).unwrap();
        let mut off = Sgd::new(cfg(0, false, schedule, Decay::Lam(0.99))).unwrap();
        // Up to and including the row the first buffer fills (target 1's:
        // present on every row); after it the fits, and so the gradients,
        // part.
        for (x, y, w) in &data {
            let (a, b) = (on.step(x, y, 1.0, *w), off.step(x, y, 1.0, *w));
            assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits());
            assert_eq!(on.w_target, off.w_target);
            assert_eq!(on.g2, off.g2);
            if !on.refit[1].open() {
                break;
            }
        }
        assert!(!on.refit[1].open(), "the refit ran");
        assert_ne!(on.beta[1], off.beta[1], "and moved the fit");
        assert_eq!(on.beta[0], off.beta[0], "and only its own target's");
    }
}

/// Saved and loaded mid-buffer, the model goes on as one that never
/// stopped, to the bit: the buffer is state.
#[test]
fn a_save_and_load_mid_buffer_goes_on_to_the_bit() {
    let data = rows(120, true);
    let c = cfg(50, true, LearningRate::AdaGrad, Decay::Lam(0.995));
    let whole = run(&mut Sgd::new(c.clone()).unwrap(), &data);
    let mut m = Sgd::new(c).unwrap();
    let mut parts = run(&mut m, &data[..30]);
    assert!(m.refit.iter().all(RefitBuf::open));
    let bytes = rmp_serde::to_vec(&m.state()).unwrap();
    let mut back = Sgd::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    parts.extend(run(&mut back, &data[30..]));
    assert_eq!(bits(&whole), bits(&parts));
}

/// A zero-weight row is clock alone (hard rule 9): not buffered and not
/// counted, so without decay a stream with such rows predicts every other
/// row as the stream without them does, to the bit -- a first row of weight
/// 0 included.
#[test]
fn zero_weight_rows_neither_count_nor_move_the_refit() {
    let with = rows(120, true);
    let without: Vec<_> = with.iter().filter(|r| r.2 > 0.0).cloned().collect();
    let c = cfg(40, false, LearningRate::Constant, Decay::Lam(1.0));
    let mut lead = vec![([0.3, 2.5], [Some(1000.0), Some(0.0)], 0.0)];
    lead.extend(with.iter().cloned());
    let a = run(&mut Sgd::new(c.clone()).unwrap(), &lead);
    let b = run(&mut Sgd::new(c).unwrap(), &without);
    let kept: Vec<Vec<f64>> = lead
        .iter()
        .zip(a)
        .filter(|(r, _)| r.2 > 0.0)
        .map(|(_, p)| p)
        .collect();
    assert_eq!(bits(&kept), bits(&b));
}

/// Under a constraint the refit's fit is projected on it, as a step is.
#[test]
fn the_refit_is_projected_on_a_constraint() {
    let data = rows(40, false);
    let mut c = cfg(30, false, LearningRate::Constant, Decay::Lam(1.0));
    c.constraint = Some(Constraint {
        lo: vec![0.0, 0.0],
        hi: vec![0.5, 0.5],
        sum: None,
    });
    let mut m = Sgd::new(c).unwrap();
    run(&mut m, &data);
    assert!(m.refit.iter().all(|b| !b.open()));
    for b in m.coefficients() {
        assert!(b[1..].iter().all(|v| (0.0..=0.5).contains(v)), "{b:?}");
    }
}

/// The setting is the Huber loss's alone.
#[test]
fn the_refit_is_refused_beside_another_loss() {
    let mut c = cfg(10, false, LearningRate::Constant, Decay::Lam(1.0));
    c.loss = SgdLoss::Squared;
    c.huber_scale = crate::HuberScale::Rms;
    let e = Sgd::new(c).unwrap_err();
    assert!(e.contains("huber_refit_rows are for loss \"huber\""), "{e}");
}

/// An infinite cut is the squared loss, so the refit is refused beside one,
/// and a cut of `inf` with the refit off is taken.
#[test]
fn the_refit_is_refused_beside_an_infinite_cut() {
    let mut c = cfg(10, false, LearningRate::Constant, Decay::Lam(1.0));
    c.loss = SgdLoss::Huber {
        delta: f64::INFINITY,
    };
    let e = Sgd::new(c.clone()).unwrap_err();
    assert!(e.contains("needs a finite huber_delta"), "{e}");
    c.refit_rows = 0;
    assert!(Sgd::new(c).is_ok());
}

/// A state's buffers are read as its configuration writes them: one a
/// target while the refit is on, each of whole rows, none while it is off.
/// One short of a target's buffer, or with buffers under a refit that is
/// off, does not load.
#[test]
fn a_state_whose_buffers_do_not_fit_does_not_load() {
    let data = rows(30, false);
    let mut m = Sgd::new(cfg(60, false, LearningRate::Constant, Decay::Lam(1.0))).unwrap();
    run(&mut m, &data);
    let loads = |m: &Sgd| {
        let bytes = rmp_serde::to_vec(&m.state()).unwrap();
        rmp_serde::from_slice::<State>(&bytes).is_ok()
    };
    assert!(loads(&m));
    let mut short = m.clone();
    short.refit.pop();
    assert!(!loads(&short), "a target's buffer missing");
    let mut off = m.clone();
    off.cfg.refit_rows = 0;
    assert!(!loads(&off), "buffers under a refit that is off");
}

/// The refit's fit under a constraint is projected on it in the
/// coordinates the coefficients are held in: while the scaler warms up
/// (a refit on the tenth row) and once they are held in the caller's units
/// (on the fortieth).
#[test]
fn the_refit_is_projected_under_the_scaler() {
    for at in [10, 40] {
        let data = rows(at as usize, false);
        let mut c = cfg(at, true, LearningRate::Constant, Decay::Lam(1.0));
        c.constraint = Some(Constraint {
            lo: vec![0.0, 0.0],
            hi: vec![0.5, 0.5],
            sum: None,
        });
        let mut m = Sgd::new(c).unwrap();
        run(&mut m, &data);
        assert!(!m.refit[1].open(), "the refit ran at {at}");
        assert_eq!(m.scaler_warming(), at == 10);
        for b in m.coefficients() {
            assert!(
                b[1..].iter().all(|v| (-1e-12..=0.5 + 1e-12).contains(v)),
                "{at}: {b:?}"
            );
        }
    }
}

/// `l2` enters the refit as the steps apply it, once a row on each slope:
/// the refit is the fixed point of the penalized weighted least squares at
/// its own Huber weights, solved here by faer's LU (`crate::oracle`) on the
/// raw normal equations, the intercept unpenalized; and it is shrunk
/// against the fit without the penalty.
#[test]
fn the_refit_takes_l2_as_the_steps_do() {
    let n = 40usize;
    let data = rows(n, false);
    let fit = |l2: f64| {
        let mut c = cfg(n as u32, false, LearningRate::Constant, Decay::Lam(1.0));
        c.l2 = l2;
        let mut m = Sgd::new(c).unwrap();
        run(&mut m, &data);
        assert!(!m.refit[1].open());
        m.coefficients()[1].clone()
    };
    let (b, free) = (fit(0.5), fit(0.0));
    assert!(
        b[1].abs() + b[2].abs() < free[1].abs() + free[2].abs(),
        "{b:?} {free:?}"
    );
    // The fixed point, target 1 (present on every row), unit weights, no decay.
    let rows: Vec<([f64; 3], f64)> = data
        .iter()
        .map(|(x, y, _)| ([1.0, x[0], x[1]], y[1].unwrap()))
        .collect();
    let r: Vec<f64> = rows
        .iter()
        .map(|(z, y)| (y - (b[0] * z[0] + b[1] * z[1] + b[2] * z[2])).abs())
        .collect();
    let s = crate::huber_refit::weighted_median(&r, &vec![1.0; n]) / crate::NORMAL_Q3;
    let u: Vec<f64> = r.iter().map(|a| (1.345 * s / a).min(1.0)).collect();
    let total: f64 = u.iter().sum();
    let mut a = vec![0.0; 9];
    let mut rhs = vec![0.0; 3];
    for ((z, y), ui) in rows.iter().zip(&u) {
        for p in 0..3 {
            rhs[p] += ui * z[p] * y;
            for q in 0..3 {
                a[p * 3 + q] += ui * z[p] * z[q];
            }
        }
    }
    for p in 1..3 {
        a[p * 3 + p] += REFIT_RIDGE * total + 0.5 * n as f64;
    }
    let again = crate::oracle::solve(&a, &rhs);
    for (p, q) in b.iter().zip(&again) {
        assert!((p - q).abs() < 1e-7 * (1.0 + q.abs()), "{b:?} {again:?}");
    }
}

/// A sparse target under an irregular clock holds a run of ageings a row
/// it waits on: past four a learned row of its count, the refit is given up,
/// its buffer freed, and the status says so; the fit is the one pass's.
#[test]
fn a_sparse_target_gives_its_refit_up_past_the_cap() {
    let mut m = Sgd::new(cfg(
        20,
        false,
        LearningRate::Constant,
        Decay::Halflife(50.0),
    ))
    .unwrap();
    let mut s = 3u64;
    for i in 0..400 {
        let x = [lcg(&mut s), lcg(&mut s)];
        // Target 0 on every tenth row; target 1 on every row.
        let y0 = (i % 10 == 0).then(|| 1.0 + x[0]);
        let d = 1.0 + 0.5 * lcg(&mut s).abs();
        m.step(&x, &[y0, Some(x[1])], d, 1.0);
    }
    let st = m.refit_status();
    assert!(st[0].done && st[0].abandoned && !st[0].failed, "{st:?}");
    assert!(st[1].done && !st[1].abandoned && st[1].passes > 0, "{st:?}");
    assert_eq!(m.refit[0].held(), 0);
}

/// Task 278: a refit that takes every one of its IRLS passes is counted in
/// `solve_failures` (`huber`'s rule, read from the target's refit status):
/// `slow_refit_stream(687)` runs to the cap of 200, trial 0's converges
/// inside it.
#[test]
fn a_refit_that_runs_to_its_cap_is_counted() {
    for (trial, capped) in [(687, true), (0, false)] {
        let rows = crate::huber_refit::slow_refit_stream(trial);
        let mut c = cfg(
            rows.len() as u32,
            false,
            LearningRate::Constant,
            Decay::Halflife(f64::INFINITY),
        );
        c.n_features = 1;
        c.n_targets = 1;
        let mut m = Sgd::new(c).unwrap();
        assert_eq!(m.solve_failures(), 0);
        for (x, y) in &rows {
            m.step(&[*x], &[Some(*y)], 1.0, 1.0);
        }
        let st = m.refit_status()[0];
        assert!(st.done && !st.failed, "trial {trial}: {st:?}");
        assert_eq!(st.ran_to_cap(), capped, "trial {trial}: {st:?}");
        assert_eq!(m.solve_failures(), u64::from(capped), "trial {trial}");
    }
}
