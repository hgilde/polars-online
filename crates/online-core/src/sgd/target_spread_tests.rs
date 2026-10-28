//! `sgd` in the target's own units, and its gate on an unstable rate
//! (`SgdCfg::target_spread`; docs/PLAN.md task 278, D1, D2, D8 and D-S):
//! the squared and Huber losses' clip and AdaGrad's epsilon scale with the
//! target, to the bit, where they were constants in its units; the
//! intercept takes the running mean's step while it is the larger, the
//! first learned row's whole residual, so a level never passes through the
//! slopes' steps; and a target whose recent steps' gains say the rate is
//! past its stability bound has its predictions withheld, by `step` and
//! `predict` alike, while its fit learns as it would have.

use super::*;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn cfg(loss: SgdLoss, schedule: LearningRate, learning_rate: f64) -> SgdCfg {
    SgdCfg {
        n_features: 3,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(300.0),
        loss,
        learning_rate,
        schedule,
        l2: 0.0,
        min_weight: 3.0,
        clip_gradient: 1e3,
        constraint: None,
        standardize: true,
        strict_binary: false,
        huber_scale: if matches!(loss, SgdLoss::Huber { .. }) {
            crate::HuberScale::Mad
        } else {
            crate::HuberScale::Rms
        },
        huber_scale_decay: None,
        refit_rows: 0,
        target_spread: true,
        mad_anchor: false,
        min_weight_holds_learning: false,
    }
}

/// One row: the features and the target.
type Row = ([f64; 3], f64);

/// Three features at their own levels and scales, a target at `level` with
/// uniform noise and, every 23rd row, an outlier of +40: no libm on the
/// path, so the bits are every platform's.
fn rows(n: usize, level: f64) -> Vec<Row> {
    let mut s = 5u64;
    (0..n)
        .map(|i| {
            let x = [
                0.5 + lcg(&mut s),
                -2.0 + 3.0 * lcg(&mut s),
                4.0 + 0.5 * lcg(&mut s),
            ];
            let out = if i % 23 == 7 { 40.0 } else { 0.0 };
            let y = level + 0.3 + 1.5 * x[0] - 0.7 * x[1] + 2.0 * x[2] + 0.5 * lcg(&mut s) + out;
            (x, y)
        })
        .collect()
}

/// The predictions and, after the last row, the coefficients of `cfg` over
/// `rows` with the target scaled by `c`.
fn run(cfg: &SgdCfg, rows: &[Row], c: f64) -> (Vec<f64>, Vec<f64>) {
    let mut m = Sgd::new(cfg.clone()).unwrap();
    let pred = rows
        .iter()
        .map(|(x, y)| m.step(x, &[Some(y * c)], 1.0, 1.0).pred[0])
        .collect();
    (pred, m.coefficients()[0].clone())
}

/// How many entries of `a` and `b` differ in their bits.
fn differ(a: &[f64], b: &[f64]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

fn scaled(v: &[f64], c: f64) -> Vec<f64> {
    v.iter().map(|f| f * c).collect()
}

/// A power of two is exact, and nothing in the target's units remains, so a
/// target scaled by `2^±20` fits as the unscaled one scaled, to the bit:
/// under the squared and Huber losses with a clip that binds (0.5 of the
/// target's spread), and under AdaGrad, whose epsilon was `1e-8` in the
/// gradient's units. Off (a state before schema 64) neither holds: the clip
/// binds on one scale and not the other, and the epsilon is 1e-8 of a
/// gradient near 1e-6.
#[test]
fn the_clip_and_adagrads_epsilon_scale_with_the_target_to_the_bit() {
    let rows = rows(600, 0.0);
    let cases = [
        ("squared", SgdLoss::Squared, LearningRate::Constant, 0.05),
        (
            "huber",
            SgdLoss::Huber { delta: 1.345 },
            LearningRate::InvScaling { power: 0.5 },
            0.1,
        ),
        ("adagrad", SgdLoss::Squared, LearningRate::AdaGrad, 0.1),
    ];
    for (what, loss, schedule, lr) in cases {
        let mut tight = cfg(loss, schedule, lr);
        tight.clip_gradient = 0.5;
        // The evidence the clip binds: without it the fit is another.
        let free = SgdCfg {
            clip_gradient: f64::INFINITY,
            ..tight.clone()
        };
        let (p, _) = run(&tight, &rows, 1.0);
        assert!(
            differ(&p, &run(&free, &rows, 1.0).0) > 0,
            "{what}: the clip never bound"
        );
        for k in [-20, 20] {
            let c = 2f64.powi(k);
            // AdaGrad's step is the rate in the coefficient's units.
            let at = |mut cfg: SgdCfg| {
                if matches!(schedule, LearningRate::AdaGrad) {
                    cfg.learning_rate = lr * c;
                }
                cfg
            };
            for base in [&tight, &free] {
                let (p0, b0) = run(base, &rows, 1.0);
                let (p0, b0) = (scaled(&p0, c), scaled(&b0, c));
                let (p1, b1) = run(&at(base.clone()), &rows, c);
                assert_eq!(differ(&p1, &p0), 0, "{what} at 2^{k}: predictions");
                assert_eq!(differ(&b1, &b0), 0, "{what} at 2^{k}: coefficients");
            }
            let old = SgdCfg {
                target_spread: false,
                ..tight.clone()
            };
            let (p0, _) = run(&old, &rows, 1.0);
            let (p1, _) = run(&at(old), &rows, c);
            assert!(
                differ(&p1, &scaled(&p0, c)) > 0,
                "{what} at 2^{k}: a state before schema 64 fitted otherwise"
            );
        }
    }
}

/// A target's first learned row sets the intercept to its own value, whatever
/// coordinates it is held in, and moves nothing else; a zero-weight row or a
/// null target before it is not that row. A level then moves the fit by
/// rounding alone. The log and logit links and a fit without an intercept
/// take no such row.
#[test]
fn a_targets_first_learned_row_sets_the_intercept() {
    for standardize in [true, false] {
        let mut m = Sgd::new(SgdCfg {
            standardize,
            min_weight: 0.0,
            ..cfg(SgdLoss::Squared, LearningRate::Constant, 0.05)
        })
        .unwrap();
        let x = [1.0, -2.0, 3.0];
        m.step(&x, &[None], 1.0, 1.0);
        m.step(&x, &[Some(5.0)], 1.0, 0.0);
        assert_eq!(m.coefficients()[0], vec![0.0; 4], "not yet a learned row");
        m.step(&x, &[Some(1000.0)], 1.0, 1.0);
        let b = m.coefficients()[0].clone();
        assert_eq!(
            &b[1..],
            &[0.0; 3],
            "standardize {standardize}: the slopes moved"
        );
        assert_eq!(b[0], 1000.0, "standardize {standardize}");
        // The next row steps: its residual is the level's no more.
        let p = m.step(&[2.0, 0.0, 1.0], &[Some(1001.0)], 1.0, 1.0).pred[0];
        assert_eq!(p, 1000.0, "standardize {standardize}");
    }
    for (what, loss, intercept) in [
        ("poisson", SgdLoss::Poisson, true),
        ("logistic", SgdLoss::Logistic, true),
        ("no intercept", SgdLoss::Squared, false),
    ] {
        let mut m = Sgd::new(SgdCfg {
            fit_intercept: intercept,
            standardize: false,
            ..cfg(loss, LearningRate::Constant, 0.05)
        })
        .unwrap();
        let mut plain = Sgd::new(SgdCfg {
            target_spread: false,
            ..m.cfg().clone()
        })
        .unwrap();
        for (x, y) in rows(5, 0.0) {
            let y = (y > 0.0) as u8 as f64;
            m.step(&x, &[Some(y)], 1.0, 1.0);
            plain.step(&x, &[Some(y)], 1.0, 1.0);
        }
        assert_eq!(m.coefficients(), plain.coefficients(), "{what}");
    }
    // The level: every residual the same but for the level's rounding.
    let base = cfg(
        SgdLoss::Squared,
        LearningRate::InvScaling { power: 1.0 },
        1.0,
    );
    let rows0 = rows(2000, 0.0);
    let rows1 = rows(2000, 1e6);
    let (p0, b0) = run(&base, &rows0, 1.0);
    let (p1, b1) = run(&base, &rows1, 1.0);
    let worst = p0
        .iter()
        .zip(&p1)
        .filter(|(a, _)| a.is_finite())
        .map(|(a, b)| (b - 1e6 - a).abs())
        .fold(0.0, f64::max);
    // A few ulps of the level a row, kept as the steps keep a residual:
    // measured 2.3e-8 (2026-10-26), against the 9.8e5 the slopes' steps
    // carried off the level before.
    assert!(worst < 1e-6, "{worst}");
    assert!((b1[0] - 1e6 - b0[0]).abs() < 1e-6);
    for j in 1..4 {
        assert!(
            (b1[j] - b0[j]).abs() < 1e-9,
            "slope {j}: {} {}",
            b1[j],
            b0[j]
        );
    }
}

/// The step gain the gate reads, by hand: `q = lr·(w·c·|z̃|² + l2)` a
/// stepped row, without a scaler `z̃ = [1, x]`, the sums decayed by
/// `2^(-1/100)`, and a prediction withheld while `Σq² > 2·Σq` from the
/// rows before. At a constant rate past the bound it withholds nearly every
/// row; under `inv_scaling` from the same rate its first rows, until the
/// rate has fallen; at a stable rate none. `predict` withholds the rows
/// `step` does.
#[test]
fn the_gate_reads_the_step_gains_mean_square_against_their_mean() {
    let rows = rows(3000, 0.0);
    let cases = [
        (
            "constant past the bound",
            LearningRate::Constant,
            0.15,
            2900,
        ),
        (
            "inv_scaling",
            LearningRate::InvScaling { power: 1.0 },
            0.15,
            1,
        ),
        ("stable", LearningRate::Constant, 0.005, 0),
    ];
    for (what, schedule, lr, at_least) in cases {
        let c = SgdCfg {
            standardize: false,
            l2: 0.001,
            min_weight: 0.0,
            ..cfg(SgdLoss::Squared, schedule, lr)
        };
        let mut m = Sgd::new(c.clone()).unwrap();
        let (mut q1, mut q2, mut w_sum) = (0.0f64, 0.0f64, 0.0f64);
        let mut withheld = 0;
        let mut released_after = false;
        for (i, (x, y)) in rows.iter().enumerate() {
            let want = q2 > 2.0 * q1;
            let ahead = m.predict(x, 1.0).pred[0];
            let pred = m.step(x, &[Some(*y)], 1.0, 1.0).pred[0];
            assert_eq!(
                ahead.to_bits(),
                pred.to_bits(),
                "{what}, row {i}: predict and step"
            );
            let gated = want;
            assert_eq!(pred.is_nan(), want, "{what}, row {i}");
            assert_eq!(m.unstable(), {
                // The row just learned joins the sums the next row reads.
                let lam = c.decay.factor(1.0);
                let lr_row = match schedule {
                    LearningRate::InvScaling { power } => lr / (1.0 + lam * w_sum).powf(power),
                    _ => lr,
                };
                if i > 0 {
                    let zz = 1.0 + x.iter().map(|v| v * v).sum::<f64>();
                    let q = lr_row * (zz + c.l2);
                    q1 = GAIN_DECAY * q1 + q;
                    q2 = GAIN_DECAY * q2 + q * q;
                }
                w_sum = lam * w_sum + 1.0;
                q2 > 2.0 * q1
            });
            withheld += usize::from(gated);
            released_after |= withheld > 0 && !gated;
        }
        println!("{what}: {withheld} rows withheld");
        assert!(withheld >= at_least, "{what}: {withheld}");
        if at_least == 0 {
            assert_eq!(withheld, 0, "{what}");
        }
        if matches!(schedule, LearningRate::InvScaling { .. }) {
            assert!(
                released_after,
                "{what}: the gate never let a row through again"
            );
        }
    }
}

/// The gate is the output's alone: a model whose gains are forced past the
/// bound learns every row as one whose gains are not -- the coefficients,
/// the residual scale and its sketch, the target's spread -- and only its
/// predictions are withheld (docs/PLAN.md task 278, D10's rule).
#[test]
fn the_gate_moves_nothing_the_fit_learns() {
    let rows = rows(400, 100.0);
    let c = cfg(
        SgdLoss::Huber { delta: 1.345 },
        LearningRate::Constant,
        0.05,
    );
    let mut free = Sgd::new(c.clone()).unwrap();
    let mut gated = Sgd::new(c).unwrap();
    gated.gain[0] = StepGain { q: 0.0, q2: 1e300 };
    let mut withheld = 0;
    for (x, y) in &rows {
        let a = free.step(x, &[Some(*y)], 1.0, 1.0).pred[0];
        let b = gated.step(x, &[Some(*y)], 1.0, 1.0).pred[0];
        withheld += usize::from(a.is_finite() && b.is_nan());
    }
    assert!(withheld > 390, "{withheld} rows withheld");
    assert_eq!(free.beta, gated.beta);
    assert_eq!((&free.sig2, &free.wsig), (&gated.sig2, &gated.wsig));
    assert_eq!(free.mad, gated.mad);
    assert_eq!(free.spread, gated.spread);
}

/// The gains and the target's spread go through a save and a load: the
/// restored model withholds and fits as the one that went on.
#[test]
fn the_gains_and_the_spread_survive_a_save() {
    let rows = rows(300, 0.0);
    let mut m = Sgd::new(SgdCfg {
        standardize: false,
        ..cfg(SgdLoss::Squared, LearningRate::Constant, 0.15)
    })
    .unwrap();
    for (x, y) in &rows[..150] {
        m.step(x, &[Some(*y)], 1.0, 1.0);
    }
    assert!(m.unstable(), "past the bound by row 150");
    let mut back = Sgd::restore(&m.state()).unwrap();
    for (x, y) in &rows[150..] {
        let a = m.step(x, &[Some(*y)], 1.0, 1.0).pred[0];
        let b = back.step(x, &[Some(*y)], 1.0, 1.0).pred[0];
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert_eq!(m.coefficients(), back.coefficients());
}

/// The state's step gains are checked against its cfg as it is read: kept
/// exactly under `target_spread` with a shared rate and a loss with a
/// curvature.
#[test]
fn a_state_whose_gains_disagree_with_its_cfg_is_refused() {
    let m = Sgd::new(cfg(SgdLoss::Squared, LearningRate::Constant, 0.05)).unwrap();
    assert_eq!(m.gain.len(), 1);
    let bytes = rmp_serde::to_vec_named(&m).unwrap();
    let mut v = rmpv::decode::read_value(&mut bytes.as_slice()).unwrap();
    let rmpv::Value::Map(entries) = &mut v else {
        panic!("a map")
    };
    let gain = entries
        .iter_mut()
        .find(|(k, _)| k.as_str() == Some("gain"))
        .unwrap();
    gain.1 = rmpv::Value::Array(vec![]);
    let mut bytes = Vec::new();
    rmpv::encode::write_value(&mut bytes, &v).unwrap();
    let err = rmp_serde::from_slice::<Sgd>(&bytes)
        .unwrap_err()
        .to_string();
    assert!(err.contains("step gains"), "{err}");
    for (what, c, kept) in [
        (
            "adagrad",
            cfg(SgdLoss::Squared, LearningRate::AdaGrad, 0.1),
            0,
        ),
        (
            "quantile",
            cfg(SgdLoss::Quantile { tau: 0.5 }, LearningRate::Constant, 0.1),
            0,
        ),
        (
            "off",
            SgdCfg {
                target_spread: false,
                ..cfg(SgdLoss::Squared, LearningRate::Constant, 0.1)
            },
            0,
        ),
        (
            "logistic",
            cfg(SgdLoss::Logistic, LearningRate::Constant, 0.1),
            1,
        ),
    ] {
        assert_eq!(Sgd::new(c).unwrap().gain.len(), kept, "{what}");
    }
}

/// The level's rule, by hand, on rows whose features are 0, so no slope
/// moves and the intercept's recursion is the whole fit: `a = w / Σw` with
/// the row in, the step `−a·d` under the squared and Huber losses while
/// `a` is above the rate or Kish's count is at most 22, `a·(y − p)` under
/// the sign-valued losses while the count is, the first row's step alone,
/// and the gradient's step after; the weights uneven, a zero weight and a
/// null target among them, which the level does not count.
#[test]
fn the_intercept_takes_the_running_means_step_while_it_is_the_larger() {
    let x = [0.0; 3];
    for (what, loss, lr) in [
        ("squared", SgdLoss::Squared, 0.01),
        ("squared at a rate past 1/22", SgdLoss::Squared, 0.2),
        ("huber", SgdLoss::Huber { delta: 1.345 }, 0.01),
        ("quantile", SgdLoss::Quantile { tau: 0.7 }, 0.01),
    ] {
        let mut m = Sgd::new(SgdCfg {
            standardize: false,
            min_weight: 0.0,
            ..cfg(loss, LearningRate::Constant, lr)
        })
        .unwrap();
        let mut s = 3u64;
        let (mut b0, mut w_sum, mut w2_sum) = (0.0f64, 0.0f64, 0.0f64);
        let (mut by_mean, mut by_rate) = (0, 0);
        for i in 0..300 {
            let w = if i % 13 == 5 {
                0.0
            } else {
                0.5 + lcg(&mut s).abs()
            };
            let y = (i % 17 != 9).then(|| 40.0 + 3.0 * lcg(&mut s));
            let scale = m.scale(0);
            let p = m.step(&x, &[y], 1.0, w).pred[0];
            assert_eq!(p.to_bits(), b0.to_bits(), "{what}, row {i}: the prediction");
            let Some(y) = y.filter(|_| w > 0.0) else {
                continue;
            };
            let first = w_sum == 0.0;
            w_sum += w;
            w2_sum += w * w;
            let a = w / w_sum;
            let young = w_sum * w_sum / w2_sum <= crate::WARMUP_ROWS;
            let d = m.dloss(p, y, scale);
            let (level, warm) = match loss {
                SgdLoss::Squared | SgdLoss::Huber { .. } => (-a * d, young || a > lr),
                _ => (a * (y - p), young),
            };
            if first || warm {
                b0 += level;
                by_mean += 1;
            } else {
                b0 -= lr * (d * w).clamp(-f64::INFINITY, f64::INFINITY);
                by_rate += 1;
            }
            assert_eq!(
                m.coefficients()[0][0].to_bits(),
                b0.to_bits(),
                "{what}, row {i}"
            );
        }
        println!("{what}: {by_mean} rows by the running mean, {by_rate} by the rate");
        assert!(by_mean > 20 && by_rate > 100, "{what}: both steps ran");
    }
}

/// Task 281 (mutants): each step gain, by hand, as a number rather than
/// the gate's verdict alone: `q = lr·(w·c·|z̃|² + l2)` with `z̃ = [1, x]`
/// and `c` the loss's curvature at the row's prediction -- 1 under the
/// squared loss, 1 inside the Huber cut and 0 outside it, `p` under the
/// Poisson link, `p(1 − p)` under the logistic -- the sums decayed by
/// `2^(-1/100)` a stepped row. The weights are uneven and `l2` is not 0,
/// so the curvature, the weight, the row's length and the ridge each move
/// `q` on their own.
#[test]
fn each_step_gain_is_the_rate_times_the_curvature_along_the_row() {
    let rows = rows(400, 0.0);
    let cases = [
        ("squared", SgdLoss::Squared),
        ("huber", SgdLoss::Huber { delta: 1.345 }),
        ("poisson", SgdLoss::Poisson),
        ("logistic", SgdLoss::Logistic),
    ];
    for (what, loss) in cases {
        let c = SgdCfg {
            standardize: false,
            l2: 0.01,
            min_weight: 0.0,
            ..cfg(loss, LearningRate::Constant, 0.002)
        };
        let lr = c.learning_rate;
        let mut m = Sgd::new(c.clone()).unwrap();
        let (mut q1, mut q2) = (0.0f64, 0.0f64);
        let (mut learned, mut inside, mut outside) = (0usize, 0usize, 0usize);
        let mut s = 17u64;
        for (i, (x, y)) in rows.iter().enumerate() {
            // A target each loss teaches from: a count, a label, or the
            // stream's own.
            let y = match loss {
                SgdLoss::Poisson => (y.abs() / 4.0).floor(),
                SgdLoss::Logistic => f64::from(*y > 10.0),
                _ => *y,
            };
            let w = 0.5 + lcg(&mut s).abs();
            let first = m.level.first().is_some_and(|lv| lv.w == 0.0);
            let p = m.link(dot(&m.beta[0], 1, x));
            let scale = m.scale(0);
            m.step(x, &[Some(y)], 1.0, w);
            // The level's first row takes the whole residual and nothing
            // else: no step, no gain.
            if first {
                continue;
            }
            let curvature = match loss {
                SgdLoss::Squared => 1.0,
                SgdLoss::Huber { delta } => {
                    let within = scale.is_none_or(|sc| (p - y).abs() <= delta * sc);
                    if within {
                        inside += 1;
                    } else {
                        outside += 1;
                    }
                    f64::from(within)
                }
                SgdLoss::Poisson => p,
                SgdLoss::Logistic => p * (1.0 - p),
                _ => unreachable!(),
            };
            let zz = 1.0 + x.iter().map(|v| v * v).sum::<f64>();
            let q = lr * (w * curvature * zz + c.l2);
            q1 = GAIN_DECAY * q1 + q;
            q2 = GAIN_DECAY * q2 + q * q;
            learned += 1;
            let g = m.gain[0];
            let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs();
            assert!(
                close(g.q, q1) && close(g.q2, q2),
                "{what}, row {i}: the gains {g:?}, by hand ({q1}, {q2})"
            );
        }
        assert!(learned > 390, "{what}: {learned} rows stepped");
        if matches!(loss, SgdLoss::Huber { .. }) {
            assert!(
                inside > 100 && outside > 5,
                "{what}: {inside} rows inside the cut, {outside} outside"
            );
        }
    }
}

/// Task 281 (mutants): at a running mean's gain exactly the rate, past the
/// warm-up, the rate's step is taken, not the mean's: the mean takes over
/// only while its gain is the larger. Every weight 2 and a rate of `1/32`:
/// the 32nd learned row's gain `2 / 64` is the rate to the bit, Kish's
/// count 32 is past 22, and the two steps part, `−d/32` against
/// `−lr·(d·w) = −d/16`. Rows whose features are 0, so the intercept is the
/// whole fit.
#[test]
fn at_a_gain_equal_to_the_rate_the_rate_steps() {
    let x = [0.0; 3];
    let lr = 1.0 / 32.0;
    let mut m = Sgd::new(SgdCfg {
        standardize: false,
        min_weight: 0.0,
        clip_gradient: f64::INFINITY,
        ..cfg(SgdLoss::Squared, LearningRate::Constant, lr)
    })
    .unwrap();
    let mut s = 9u64;
    let mut b0 = 0.0f64;
    for n in 1..=40u32 {
        let y = 40.0 + 3.0 * lcg(&mut s);
        let p = m.step(&x, &[Some(y)], 1.0, 2.0).pred[0];
        let d = p - y;
        let gain = 2.0 / (2.0 * f64::from(n));
        if n == 1 {
            b0 = y;
        } else if n <= 22 || gain > lr {
            b0 += -gain * d;
        } else {
            b0 += -(lr * (d * 2.0));
        }
        if n == 32 {
            assert_eq!(gain, lr, "the tie is exact");
        }
        assert_eq!(m.coefficients()[0][0].to_bits(), b0.to_bits(), "row {n}");
    }
}

/// Task 281 (mutants): a weight so small that its square underflows to 0
/// leaves `Σw²` at 0, and Kish's count is then read as 0 -- the level still
/// warming up -- not `0/0`: the intercept takes the running mean's step,
/// `−(w/Σw)·d`, where a NaN count would hand the row to the rate, whose
/// step at such a weight is nothing.
#[test]
fn a_weight_whose_square_underflows_counts_as_a_young_level() {
    let x = [0.0; 3];
    let w = 1e-170;
    let mut m = Sgd::new(SgdCfg {
        standardize: false,
        min_weight: 0.0,
        // A rate above every gain here, so the count alone keeps the
        // level warm.
        ..cfg(SgdLoss::Squared, LearningRate::Constant, 0.9)
    })
    .unwrap();
    let mut b0 = 0.0f64;
    let mut w_sum = 0.0f64;
    for (n, y) in [40.0, 44.0, 38.0, 41.0].into_iter().enumerate() {
        let p = m.step(&x, &[Some(y)], 1.0, w).pred[0];
        w_sum += w;
        if n == 0 {
            b0 = y;
        } else {
            b0 += -(w / w_sum) * (p - y);
        }
        assert_eq!(m.level[0].w2, 0.0, "the square underflowed");
        assert_eq!(m.coefficients()[0][0].to_bits(), b0.to_bits(), "row {n}");
    }
}

/// Task 281 (mutants): `unstable_targets` is each target's own verdict: a
/// target at a rate past its bound and one that never learned (its target
/// always null) read `[true, false]`; a fresh model `[false, false]`; a
/// model that keeps no gains (AdaGrad's per-slot rate) nothing.
#[test]
fn each_target_has_its_own_verdict() {
    let rows = rows(300, 0.0);
    let mut m = Sgd::new(SgdCfg {
        n_targets: 2,
        standardize: false,
        ..cfg(SgdLoss::Squared, LearningRate::Constant, 0.15)
    })
    .unwrap();
    assert_eq!(m.unstable_targets(), vec![false, false]);
    for (x, y) in &rows {
        m.step(x, &[Some(*y), None], 1.0, 1.0);
    }
    assert_eq!(m.unstable_targets(), vec![true, false]);
    assert!(m.unstable());
    let ada = Sgd::new(cfg(SgdLoss::Squared, LearningRate::AdaGrad, 0.1)).unwrap();
    assert!(ada.unstable_targets().is_empty());
}

/// Task 281 (mutants): AdaGrad's step is 0 where its denominator is: under
/// `target_spread` the epsilon is 0 before the target has a spread, and a
/// gradient below `1e-162` has a square that underflows to 0, so the
/// first row's slope steps nothing, where `lr / 0 · g` would make it
/// infinite.
#[test]
fn adagrads_step_is_nothing_where_its_denominator_is_zero() {
    let mut m = Sgd::new(SgdCfg {
        n_features: 1,
        fit_intercept: false,
        standardize: false,
        min_weight: 0.0,
        ..cfg(SgdLoss::Squared, LearningRate::AdaGrad, 0.1)
    })
    .unwrap();
    assert_eq!(m.unit(0), 0.0, "no spread yet: the epsilon is 0");
    m.step(&[1.0], &[Some(1e-170)], 1.0, 1.0);
    assert_eq!(m.g2[0][0], 0.0, "the gradient's square underflowed");
    assert_eq!(m.coefficients()[0], vec![0.0]);
}
