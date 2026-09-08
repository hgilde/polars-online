//! Prints every prediction and the final coefficients, to the bit, for a
//! grid of `sgd` configurations over one deterministic stream. Two builds of
//! the crate that print the same text compute the same numbers.
use online_core::{Decay, LearningRate, OnlineModel, Sgd, SgdCfg, SgdLoss};

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// One stream row: features, targets, clock step, weight.
type Row = (Vec<f64>, Vec<Option<f64>>, f64, f64);

fn main() {
    let (k, m, n) = (7usize, 2usize, 400usize);
    let mut seed = 12345u64;
    let rows: Vec<Row> = (0..n)
        .map(|r| {
            let x: Vec<f64> = (0..k)
                .map(|i| 10f64.powi(i as i32 % 3 - 1) * lcg(&mut seed) + 0.5 * i as f64)
                .collect();
            let y: Vec<Option<f64>> = (0..m)
                .map(|j| {
                    if r % 37 == 5 && j == 0 {
                        None
                    } else {
                        Some(x[j] * 2.0 - x[(j + 1) % k] + 0.1 * lcg(&mut seed))
                    }
                })
                .collect();
            let d_clock = if r % 23 == 0 { 3.0 } else { 1.0 };
            let w = match r % 11 {
                0 => 0.0,
                1 => 2.5,
                _ => 1.0,
            };
            (x, y, d_clock, w)
        })
        .collect();
    for add_intercept in [true, false] {
        for scale in [false, true] {
            for (sname, schedule) in [
                ("const", LearningRate::Constant),
                ("inv", LearningRate::InvScaling { power: 0.3 }),
                ("adagrad", LearningRate::AdaGrad),
            ] {
                for (lname, loss) in [
                    ("sq", SgdLoss::Squared),
                    ("huber", SgdLoss::Huber { delta: 0.5 }),
                ] {
                    for l2 in [0.0, 1e-3] {
                        let cfg = SgdCfg {
                            n_features: k,
                            n_targets: m,
                            add_intercept,
                            decay: Decay::Halflife(50.0),
                            loss,
                            learning_rate: 0.02,
                            schedule,
                            l2,
                            min_periods: 5.0,
                            scale_features: scale,
                            clip_gradient: 5.0,
                            constraint: None,
                        };
                        let mut model = Sgd::new(cfg).unwrap();
                        let mut acc = 0u64;
                        let mut last = Vec::new();
                        for (x, y, d, w) in &rows {
                            let s = model.step(x, y, *d, *w);
                            let p = model.predict(x, *d);
                            for v in s.pred.iter().chain(&p.pred) {
                                acc = acc.wrapping_mul(31).wrapping_add(v.to_bits());
                            }
                            last = s.pred.clone();
                        }
                        let coefs = model.coefficients();
                        println!(
                            "int={add_intercept} scale={scale} {sname} {lname} l2={l2}: hash={acc:016x} last={last:?} coef0={:?}",
                            coefs[0]
                        );
                    }
                }
            }
        }
    }
}
