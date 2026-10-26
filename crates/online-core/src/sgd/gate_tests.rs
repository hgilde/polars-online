//! `min_weight` withholds `sgd`'s output alone (docs/PLAN.md task 278,
//! D10): under the Huber loss the residual scale learns from the prediction
//! the fit made, whether or not the gate withholds it, so the coefficients,
//! the scale and the refit are the same at any `min_weight`. Under the gate
//! that held learning back -- a state written before schema 64
//! (`SgdCfg::min_weight_holds_learning`) -- they part, which is the
//! evidence the stream reaches the path.

use super::*;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn cfg(scale: crate::HuberScale, refit_rows: u32, holds: bool) -> SgdCfg {
    SgdCfg {
        n_features: 2,
        n_targets: 2,
        fit_intercept: true,
        decay: Decay::Halflife(40.0),
        loss: SgdLoss::Huber { delta: 1.345 },
        learning_rate: 0.05,
        schedule: LearningRate::AdaGrad,
        l2: 0.0,
        min_weight: 0.0,
        clip_gradient: 1e3,
        constraint: None,
        standardize: true,
        strict_binary: false,
        huber_scale: scale,
        huber_scale_decay: None,
        refit_rows,
        min_weight_holds_learning: holds,
        // The target's own units and the unstable rate's gate, as a spec
        // builds them: the scale's learning composes with both.
        target_spread: true,
    }
}

/// Per row the coefficients' and the scale's bits and the output; a
/// one-sided outlier stream with a target joining at row 30, uneven and
/// zero weights, and a gap at row 200 under every `min_weight` here.
fn run(c: &SgdCfg, min_weight: f64) -> (Vec<Vec<u64>>, Vec<Vec<f64>>) {
    let mut m = Sgd::new(SgdCfg {
        min_weight,
        ..c.clone()
    })
    .unwrap();
    let mut s = 29u64;
    let (mut fit, mut out) = (Vec::new(), Vec::new());
    for i in 0..400 {
        let x = [lcg(&mut s), 1.0 + lcg(&mut s)];
        let wild = if x[0] > 0.3 && lcg(&mut s) > 0.5 {
            20.0
        } else {
            0.0
        };
        let y0 = 4.0 + 2.0 * x[0] - x[1] + 0.3 * lcg(&mut s) + wild;
        let y1 = (i >= 30).then(|| -1.0 + x[1] + 0.2 * lcg(&mut s) - wild);
        let d = match i {
            0 => 0.0,
            200 => 150.0,
            _ => 1.0,
        };
        let w = if i % 11 == 5 {
            0.0
        } else {
            0.5 + lcg(&mut s).abs()
        };
        out.push(m.step(&x, &[Some(y0), y1], d, w).pred);
        let mut bits: Vec<u64> = m
            .coefficients()
            .iter()
            .flatten()
            .map(|v| v.to_bits())
            .collect();
        bits.extend(m.sigma2().iter().map(|v| v.to_bits()));
        if let Some(q) = m.mad_sketch().first() {
            bits.push(q.get(0).unwrap_or(f64::NAN).to_bits());
        }
        fit.push(bits);
    }
    (fit, out)
}

#[test]
fn min_weight_withholds_the_output_alone() {
    for (name, scale, refit) in [
        ("mad, refit", crate::HuberScale::Mad, 60),
        ("rms", crate::HuberScale::Rms, 0),
    ] {
        let c = cfg(scale, refit, false);
        let (base, out0) = run(&c, 0.0);
        for mw in [3.0, 25.0] {
            let (fit, out) = run(&c, mw);
            assert_eq!(fit, base, "{name}, min_weight {mw}: the fit");
            let mut withheld = 0;
            for (i, (a, b)) in out0.iter().zip(&out).enumerate() {
                for (p, q) in a.iter().zip(b) {
                    if q.is_nan() && p.is_finite() {
                        withheld += 1;
                    } else {
                        assert_eq!(p.to_bits(), q.to_bits(), "{name}, {mw}, row {i}");
                    }
                }
            }
            println!("{name}, min_weight {mw}: {withheld} outputs withheld");
            assert!(withheld > 0, "{name}, {mw}: the gate withheld nothing");
        }
        let held = cfg(scale, refit, true);
        assert_ne!(
            run(&held, 25.0).0,
            run(&held, 0.0).0,
            "{name}: the gate that held learning back moved the fit"
        );
    }
}
