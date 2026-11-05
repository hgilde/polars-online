//! The MAD scale's equivariance under a scale and a shift of the target
//! (task 323). Each stream is `y = L + 2x + ε`, `x` and `ε` standard
//! normal, at levels `L` of 1, 1e3, 1e6 and 1e9, with the row whose
//! residual the scale takes first set to `1e-6` from its prediction: the
//! quiet start that defeated an anchor chosen from the first residual.

use super::*;

/// A standard normal by Box-Muller from an LCG: the two runs compared are
/// on one platform, so libm's last bit is the same in both.
fn normal(state: &mut u64) -> f64 {
    let mut u = || {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let (a, b) = (u(), u());
    (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
}

fn cfg(loss: RobustLoss, refit_rows: u32) -> RobustCfg {
    RobustCfg {
        n_features: 1,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(500.0),
        loss,
        ridge: 1e-6,
        standardize: false,
        min_weight: 0.0,
        solve_every: 0.0,
        max_rows_between_solves: u32::MAX,
        solve_share: Some(crate::DEFAULT_SOLVE_SHARE),
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

/// `m` as a build of schema 64 saved it before any row, restored: each
/// sketch awaiting its first value as its anchor, each buffer holding no
/// `|e|`, so it runs by that build's rule (the first-value anchor).
fn as_64(m: Robust) -> Robust {
    fn old(v: &mut rmpv::Value) {
        match v {
            rmpv::Value::Map(m) => {
                for (k, v) in m.iter_mut() {
                    match k.as_str() {
                        Some("anchor") if v.as_f64() == Some(1.0) => *v = rmpv::Value::from(0.0),
                        Some("resid") => *v = rmpv::Value::Nil,
                        _ => old(v),
                    }
                }
            }
            rmpv::Value::Array(a) => a.iter_mut().for_each(old),
            _ => {}
        }
    }
    let mut s = m.state();
    s.schema_version = 64;
    let bytes = rmp_serde::to_vec_named(&s).unwrap();
    let mut v = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
    old(&mut v);
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &v).unwrap();
    Robust::restore(&rmp_serde::from_slice(&out).unwrap()).unwrap()
}

/// The stream at level `level`: `x`, and `y` with the first scored row's
/// target `1e-6` above its prediction.
fn stream(seed: u64, n: usize, level: f64, loss: RobustLoss, refit: u32) -> (Vec<f64>, Vec<f64>) {
    let mut st = 0x5eed_0323 + seed * 7919;
    let x: Vec<f64> = (0..n).map(|_| normal(&mut st)).collect();
    let mut y: Vec<f64> = (0..n)
        .map(|i| level + 2.0 * x[i] + normal(&mut st))
        .collect();
    let mut m = Robust::new(cfg(loss, refit)).unwrap();
    for i in 0..n {
        let p = m.predict(&[x[i]], 1.0).pred[0];
        if p.is_finite() {
            y[i] = p + 1e-6;
            break;
        }
        m.step(&[x[i]], &[Some(y[i])], 1.0, 1.0);
    }
    (x, y)
}

/// Each row's residual scale after it, and its slope.
fn run(mut m: Robust, x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (mut s, mut b) = (Vec::new(), Vec::new());
    for i in 0..x.len() {
        m.step(&[x[i]], &[Some(y[i])], 1.0, 1.0);
        s.push(m.residual_scale(0).unwrap_or(f64::NAN));
        b.push(m.coefficients().map_or(f64::NAN, |c| c[0][1]));
    }
    (s, b)
}

/// The largest `|got − c · base| / (c · unit)` over the rows where both are
/// finite: in units of `unit` (`None`: of `base` itself, relative).
fn gap_in(base: &[f64], got: &[f64], c: f64, unit: Option<f64>) -> f64 {
    base.iter()
        .zip(got)
        .filter(|(a, b)| a.is_finite() && b.is_finite() && **a != 0.0)
        .map(|(a, b)| (b - c * a).abs() / (c * unit.unwrap_or(a.abs())))
        .fold(0.0, f64::max)
}

/// The largest `|got / (c · base) − 1|` over the rows where both are finite.
fn gap(base: &[f64], got: &[f64], c: f64) -> f64 {
    base.iter()
        .zip(got)
        .filter(|(a, b)| a.is_finite() && b.is_finite() && **a != 0.0)
        .map(|(a, b)| (b / (c * a) - 1.0).abs())
        .fold(0.0, f64::max)
}

/// The five seeds of every case, at `n` rows: each stream with its
/// largest target `Λ`, and the runs of `make` on it, on the same stream
/// times 3 and 2, and on it shifted by `0.75·L`.
struct Runs {
    big: f64,
    base: (Vec<f64>, Vec<f64>),
    times3: (Vec<f64>, Vec<f64>),
    times2: (Vec<f64>, Vec<f64>),
    shifted: (Vec<f64>, Vec<f64>),
}

fn runs(seed: u64, level: f64, loss: RobustLoss, refit: u32, old: bool) -> Runs {
    let n = 1500;
    let (x, y) = stream(seed, n, level, loss, refit);
    let make = || {
        let m = Robust::new(cfg(loss, refit)).unwrap();
        if old { as_64(m) } else { m }
    };
    let scaled = |f: &dyn Fn(f64) -> f64| -> Vec<f64> { y.iter().map(|v| f(*v)).collect() };
    Runs {
        big: y.iter().fold(0.0f64, |a, v| a.max(v.abs())),
        base: run(make(), &x, &y),
        times3: run(make(), &x, &scaled(&|v| 3.0 * v)),
        times2: run(make(), &x, &scaled(&|v| 2.0 * v)),
        shifted: run(make(), &x, &scaled(&|v| v + 0.75 * level)),
    }
}

/// The bound on a gap from rounding at the targets' size `Λ`, in units of
/// the noise's spread `σ = 1`: `K · u · Λ / σ`.
///
/// `K` counts the roundings at `Λ`, by the operations a gap passes
/// through. A residual `y − p` carries the target's own (the factor or the
/// shift, 1), the prediction's (the intercept's term, the slope's product
/// and their sum, 3) and the subtraction's (1): at most 5 `uΛ`, its terms
/// at the level, the noise's part smaller. The fit's moments are
/// exponentially weighted means of the rows' deviations, each formed with
/// at most a rounding at `Λ` and the mean of the targets compensated, so
/// the fit carries a weighted average of per-row roundings, not their sum:
/// a few `uΛ` in its intercept, the same over `Var x = 1` in its slope.
/// The scale is an order statistic of the residuals at the same weights
/// -- the exact median while the refit waits, and after it a sketch whose
/// buckets are in units of the converged residuals' median, which regroup
/// a residual only within its rounding of an edge -- so its gap is at most
/// the largest residual's, 8 `uΛ` with the fit's. Huber's weight `min(1,
/// δs/|e|)` and the quantile's band `ε s` pass a relative gap in `s` or `e`
/// on to the fit multiplied by at most `max|e·x| / (δ s Var x)`, under 24
/// on these normal streams (`|x|`, `|ε| < 4.5` over 1,500 rows, `δ s ≥
/// 1.345 · 0.67`). So `8 · 24 = 192`; `K = 1024` leaves a margin of five.
/// Measured over the cases here, the worst was 178 (the shift at a level
/// of `1e9`, `huber`'s scale), and 10 under a factor of 3.
const K: f64 = 1024.0;

/// With the refit on, a stream and the same stream times 3, and shifted by
/// `0.75·L`, give scales and slopes within `K · u · Λ / σ` of each other
/// ([`K`] has the operation count) on every row -- through the warm-up,
/// whose MAD is read exactly from the buffered `|e|`, the refit at row
/// 500, and the rows after it, the sketch anchored at the converged
/// residuals' median -- at levels from 1 to `1e9`, the first residual the
/// scale takes `1e-6` of the noise: the quiet start that an anchor chosen
/// from the first residual could not survive (at `1e6` the scale stood a
/// bucket apart, `1.2e-2`, for the life of the sketch). A power of two
/// scales both to the bit.
#[test]
fn the_scale_and_the_fit_move_with_the_target_by_rounding() {
    let u = f64::EPSILON / 2.0;
    for loss in [
        RobustLoss::Huber { delta: 1.345 },
        RobustLoss::Quantile { tau: 0.5 },
    ] {
        for level in [1.0, 1e3, 1e6, 1e9] {
            for seed in 0..5 {
                let r = runs(seed, level, loss, 500, false);
                let bound = K * u * r.big;
                let (s0, b0) = &r.base;
                for (what, (s, b), c) in [("x3", &r.times3, 3.0), ("shift", &r.shifted, 1.0)] {
                    let gs = gap_in(s0, s, c, Some(1.0));
                    let gb = gap_in(b0, b, c, Some(1.0));
                    assert!(
                        gs <= bound && gb <= bound,
                        "{loss:?} L {level:e} seed {seed} {what}: scale {gs:e}, slope {gb:e}, bound {bound:e}"
                    );
                }
                let (s2, b2) = &r.times2;
                assert_eq!(gap(s0, s2, 2.0) + gap(b0, b2, 2.0), 0.0, "{loss:?} x2");
            }
        }
    }
}

/// With the refit off nothing is held and no anchor is chosen: the sketch's
/// grid is fixed at 1, so a power of two scales the scale and the fit to
/// the bit, and a factor of 3 moves `huber`'s scale by up to one bucket,
/// `e^(1/64) − 1` relative (each read within `α = tanh(1/128)` of the exact
/// median, and `(1 + α)/(1 − α) = e^(1/64)`), plus the residuals' rounding
/// at the level ([`K`]). `quantile`'s band is drawn in the scale, so a
/// bucket's move in the scale moves the rows it holds and the fit with
/// them: there a factor of 3 moved the scale by up to 2.2e-2 after row 500
/// and its slope by 4.2e-3, and is held to the power of two alone.
#[test]
fn without_the_refit_the_scale_moves_by_a_bucket_at_most() {
    let u = f64::EPSILON / 2.0;
    let bucket = (1.0f64 / 64.0).exp_m1();
    for level in [1.0, 1e3, 1e6] {
        for seed in 0..5 {
            for loss in [
                RobustLoss::Huber { delta: 1.345 },
                RobustLoss::Quantile { tau: 0.5 },
            ] {
                let r = runs(seed, level, loss, 0, false);
                let (s0, b0) = &r.base;
                let (s2, b2) = &r.times2;
                assert_eq!(gap(s0, s2, 2.0) + gap(b0, b2, 2.0), 0.0, "{loss:?} x2");
                if matches!(loss, RobustLoss::Huber { .. }) {
                    let (s3, _) = &r.times3;
                    for (a, b) in s0
                        .iter()
                        .zip(s3)
                        .filter(|(a, b)| a.is_finite() && b.is_finite())
                    {
                        assert!(
                            (b - 3.0 * a).abs() <= 3.0 * (a * bucket + K * u * r.big),
                            "L {level:e} seed {seed}: {b} against 3 x {a}"
                        );
                    }
                }
            }
        }
    }
}

/// The first-value anchor of a state saved before schema 65, which such a
/// state keeps (the activation of the tests above): at a level of `1e6`,
/// its first residual `1e-6`, the scale of a stream and of the stream
/// times 3 part by a hundred times [`K`]'s bound and more on every seed
/// (measured: 1.1e-4 to a bucket, 1.2e-2, of the noise; the bound is
/// 1.1e-7), where the rule
/// since keeps them within it.
#[test]
fn the_first_value_anchor_moves_the_scale_far_past_the_bound() {
    let u = f64::EPSILON / 2.0;
    for seed in 0..5 {
        let r = runs(seed, 1e6, RobustLoss::Huber { delta: 1.345 }, 500, true);
        let g = gap_in(&r.base.0, &r.times3.0, 3.0, Some(1.0));
        assert!(g > 100.0 * K * u * r.big, "seed {seed}: {g:e}");
    }
}
