//! The fit of a standardizing model under a moving scaler (docs/PLAN.md task
//! 206; review round 5, G1): `sgd`, `pa` and `kalman` under `standardize`.
//!
//! `sgd` and `pa` run their design from before task 206 while their scaler
//! warms up (`Warmup`, 22 rows by Kish's count), and from the row after
//! that hold their fit in the caller's units, so that the scaler's moving
//! moves no prediction. `kalman` holds `b` and `P` in the coordinates of an
//! anchor and follows every move of the moments exactly from its first row,
//! with no warm-up (docs/PLAN.md task 211). Held here: `sgd`'s and `pa`'s
//! rows up to the switch are the old build's to the bit; after it, and for
//! `kalman` throughout, rows that move only the scaler leave every
//! prediction and coefficient where it was; a power of two in the features'
//! units changes no bit; and a save at, before or after the switch goes on
//! to the bit.
//!
//! Every stream here decays by a literal factor at unit clock steps, so no
//! call into the platform's libm is on a path a bit is pinned on.

use online_core::*;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

const DECAY: Decay = Decay::Lam(0.95);

fn sgd(loss: SgdLoss, fit_intercept: bool, constraint: Option<Constraint>) -> SgdCfg {
    SgdCfg {
        n_features: 2,
        n_targets: 2,
        fit_intercept,
        decay: DECAY,
        loss,
        learning_rate: 0.05,
        schedule: LearningRate::Constant,
        l2: 0.0,
        min_weight: 3.0,
        clip_gradient: 1e3,
        constraint,
        standardize: true,
        strict_binary: false,
    }
}

fn pa(mode: PaMode, fit_intercept: bool, constraint: Option<Constraint>) -> PaCfg {
    PaCfg {
        n_features: 2,
        n_targets: 2,
        fit_intercept,
        decay: DECAY,
        mode,
        c: 1.0,
        eps: 0.05,
        min_weight: 3.0,
        constraint,
        standardize: true,
    }
}

fn kalman(fit_intercept: bool, share_p: bool) -> KalmanCfg {
    KalmanCfg {
        n_features: 2,
        n_targets: 2,
        fit_intercept,
        decay: DECAY,
        half_life: vec![30.0],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p,
        min_weight: 3.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: true,
    }
}

fn simplex() -> Option<Constraint> {
    Some(Constraint {
        lo: vec![0.0, 0.0],
        hi: vec![f64::INFINITY, f64::INFINITY],
        sum: Some(1.5),
    })
}

/// One model behind the three surfaces the tests read.
trait Fit {
    fn step(&mut self, x: &[f64], y: &[Option<f64>], d: f64, w: f64) -> Vec<f64>;
    fn predict(&self, x: &[f64]) -> Vec<f64>;
    fn coef(&self) -> Vec<Vec<f64>>;
    fn save(&self) -> Vec<u8>;
    fn load(&self, bytes: &[u8]) -> Box<dyn Fit>;
}

macro_rules! fit {
    ($ty:ty) => {
        impl Fit for $ty {
            fn step(&mut self, x: &[f64], y: &[Option<f64>], d: f64, w: f64) -> Vec<f64> {
                OnlineModel::step(self, x, y, d, w).pred
            }
            fn predict(&self, x: &[f64]) -> Vec<f64> {
                OnlineModel::predict(self, x, 1.0).pred
            }
            fn coef(&self) -> Vec<Vec<f64>> {
                self.coefficients()
            }
            fn save(&self) -> Vec<u8> {
                rmp_serde::to_vec_named(&self.state()).unwrap()
            }
            fn load(&self, bytes: &[u8]) -> Box<dyn Fit> {
                let s: State = rmp_serde::from_slice(bytes).unwrap();
                Box::new(<$ty>::restore(&s).unwrap())
            }
        }
    };
}
fit!(Sgd);
fit!(Pa);
fit!(Kalman);

/// A case's builder.
type Build = Box<dyn Fn() -> Box<dyn Fit>>;

/// Every case: a name, a builder, and whether its rows carry uneven weights
/// (0, 0.5, 1 and 2.5, which Kish's count reads as fewer rows) and a null
/// target one row in seven.
fn cases() -> Vec<(&'static str, Build, bool)> {
    vec![
        (
            "sgd squared",
            Box::new(|| Box::new(Sgd::new(sgd(SgdLoss::Squared, true, None)).unwrap())),
            false,
        ),
        (
            "sgd huber adagrad l2, weighted",
            Box::new(|| {
                let mut c = sgd(SgdLoss::Huber { delta: 1.345 }, true, None);
                c.schedule = LearningRate::AdaGrad;
                c.l2 = 0.01;
                Box::new(Sgd::new(c).unwrap())
            }),
            true,
        ),
        (
            "sgd simplex, weighted",
            Box::new(|| Box::new(Sgd::new(sgd(SgdLoss::Squared, true, simplex())).unwrap())),
            true,
        ),
        (
            "sgd through the origin",
            Box::new(|| Box::new(Sgd::new(sgd(SgdLoss::Squared, false, None)).unwrap())),
            true,
        ),
        (
            "pa1",
            Box::new(|| Box::new(Pa::new(pa(PaMode::Pa1, true, None)).unwrap())),
            false,
        ),
        (
            "pa, weighted",
            Box::new(|| Box::new(Pa::new(pa(PaMode::Pa, true, None)).unwrap())),
            true,
        ),
        (
            "pa2 simplex, weighted",
            Box::new(|| Box::new(Pa::new(pa(PaMode::Pa2, true, simplex())).unwrap())),
            true,
        ),
        (
            "pa1 through the origin",
            Box::new(|| Box::new(Pa::new(pa(PaMode::Pa1, false, None)).unwrap())),
            true,
        ),
        (
            "kalman",
            Box::new(|| Box::new(Kalman::new(kalman(true, false)).unwrap())),
            false,
        ),
        (
            "kalman share_p, weighted",
            Box::new(|| Box::new(Kalman::new(kalman(true, true)).unwrap())),
            true,
        ),
        (
            "kalman through the origin, weighted",
            Box::new(|| Box::new(Kalman::new(kalman(false, false)).unwrap())),
            true,
        ),
    ]
}

/// A row of the stream: features at a level and in units of their own, two
/// targets, the weight. `c` scales the features.
fn row(s: &mut u64, i: usize, weighted: bool, c: f64) -> ([f64; 2], [Option<f64>; 2], f64) {
    let x = [3.0 + 2.0 * lcg(s), 0.01 * lcg(s)];
    let y0 = 1.0 + 2.0 * x[0] - 50.0 * x[1] + 0.2 * lcg(s);
    let y1 = -0.5 + 0.7 * x[0] + 0.1 * lcg(s);
    let null = weighted && i % 7 == 3;
    let w = match (weighted, i % 5) {
        (false, _) => 1.0,
        (true, 1) => 2.5,
        (true, 2) => 0.5,
        (true, 3) if i > 0 => 0.0,
        _ => 1.0,
    };
    ([c * x[0], c * x[1]], [(!null).then_some(y0), Some(y1)], w)
}

/// The warm-up's length, `WARMUP_ROWS`, written out so that this file's
/// first test also compiles against the build before task 206, which made
/// its digests.
const N: f64 = 22.0;

/// The row on which the warm-up ends: the first after whose scaler update
/// Kish's count of the weights, undecayed, reaches 22.
fn switch_row(weighted: bool) -> usize {
    let (mut s, mut w1, mut w2) = (5u64, 0.0f64, 0.0f64);
    for i in 0.. {
        let (_, _, w) = row(&mut s, i, weighted, 1.0);
        w1 += w;
        w2 += w * w;
        if w2 > 0.0 && w1 * w1 / w2 >= N {
            return i;
        }
    }
    unreachable!()
}

fn fnv(h: &mut u64, v: f64) {
    // NaN in any form is one NaN.
    let bits = if v.is_nan() {
        f64::NAN.to_bits()
    } else {
        v.to_bits()
    };
    for byte in bits.to_le_bytes() {
        *h = (*h ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
    }
}

/// Up to the row that ends the warm-up, every prediction and every
/// coefficient is the build's before task 206, to the bit: the digests were
/// that build's (`ed76944`), over every row from the first to the switch,
/// both included. The switch row's predictions are made before the switch,
/// and its coefficients are the old read-out, which the switch keeps.
#[test]
fn the_rows_up_to_the_switch_are_the_old_builds_to_the_bit() {
    // Printed by this test on `ed76944`, the build before task 206.
    const DIGESTS: &[(&str, u64)] = &[
        ("sgd squared", 0xbad6_0e8b_d634_02ce),
        ("sgd huber adagrad l2, weighted", 0xaf01_4ed8_2f48_62aa),
        ("sgd simplex, weighted", 0xa800_dd9b_d794_8e88),
        ("sgd through the origin", 0x307f_85c7_ceb1_14a6),
        ("pa1", 0x8b36_e80e_36eb_165c),
        ("pa, weighted", 0xb54a_2797_8e0e_970b),
        ("pa2 simplex, weighted", 0x8958_0977_2e27_2aec),
        ("pa1 through the origin", 0x77e8_1c5a_5178_8230),
    ];
    let mut got = Vec::new();
    // `kalman` has no warm-up since task 211, and its first rows moved
    // with it: each feature's prior waits for its scale.
    for (name, build, weighted) in cases()
        .into_iter()
        .filter(|(name, _, _)| !name.starts_with("kalman"))
    {
        let mut m = build();
        let last = switch_row(weighted);
        let mut s = 5u64;
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for i in 0..=last {
            let (x, y, w) = row(&mut s, i, weighted, 1.0);
            for p in m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, w) {
                fnv(&mut h, p);
            }
            for b in m.coef().iter().flatten() {
                fnv(&mut h, *b);
            }
        }
        println!("(\"{name}\", {h:#018x}), // rows 0..={last}");
        got.push((name, h));
    }
    assert_eq!(got.len(), DIGESTS.len());
    for ((name, h), (want_name, want)) in got.iter().zip(DIGESTS) {
        assert_eq!(name, want_name);
        assert_eq!(h, want, "{name}: the rows up to the switch moved");
    }
}

/// Past the switch, rows that move only the scaler -- every target null,
/// the features at another level and spread -- leave the prediction for a
/// fixed row and the coefficients where they were: to the bit for `sgd` and
/// `pa`, whose fit is in the caller's units and reads no moment, and for
/// `kalman`, whose `b` and `P` are held at an anchor and re-mapped when the
/// moments drift from it, to rounding: within 1e-12 of `1 + |p|`. Read
/// through the moments
/// as they stood, as before task 206, the prediction moved with them though
/// no step was taken: on that build `sgd squared` predicted the fixed row
/// at 7.49 before the rows and 5.35 after them.
#[test]
fn past_the_switch_the_scaler_moves_and_the_fit_does_not() {
    let mut worst_kalman = 0.0f64;
    for (name, build, weighted) in cases() {
        let mut m = build();
        let mut s = 11u64;
        for i in 0..200 {
            let (x, y, w) = row(&mut s, i, weighted, 1.0);
            m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, w);
        }
        let fixed = [3.3, -0.004];
        let (pred, coef) = (m.predict(&fixed), m.coef());
        for _ in 0..30 {
            let x = [8.0 + 5.0 * lcg(&mut s), 0.5 + 0.2 * lcg(&mut s)];
            m.step(&x, &[None, None], 1.0, 1.0);
        }
        let (after, coef_after) = (m.predict(&fixed), m.coef());
        assert!(pred.iter().all(|p| p.is_finite()), "{name}: {pred:?}");
        if name.starts_with("kalman") {
            let rel = |a: f64, b: f64| (a - b).abs() / (1.0 + b.abs());
            for (a, b) in after.iter().zip(&pred) {
                worst_kalman = worst_kalman.max(rel(*a, *b));
            }
            for (a, b) in coef_after.iter().flatten().zip(coef.iter().flatten()) {
                worst_kalman = worst_kalman.max(rel(*a, *b));
            }
        } else {
            let bits = |v: &[f64]| v.iter().map(|p| p.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&after),
                bits(&pred),
                "{name}: the prediction moved with the scaler: {pred:?} -> {after:?}"
            );
            assert_eq!(
                coef_after, coef,
                "{name}: the coefficients moved with the scaler"
            );
        }
    }
    println!("kalman: worst relative move {worst_kalman:.2e}");
    assert!(worst_kalman <= 1e-12, "kalman: moved by {worst_kalman:e}");
}

/// Features scaled by a power of two are read to the same standardized
/// rows, so every prediction keeps its bits before the switch and after it,
/// and a slope scales by the inverse, exactly: U2's reason for
/// standardizing (docs/PLAN.md task 195) still holds of a fit held in the
/// caller's units, whose map divides by a scale and multiplies by a mean
/// that scale with the features. A constraint's bounds are the caller's, in
/// the features' units, so the constrained cases sit this one out. `kalman`
/// sat it out until task 211: it standardizes against the moments before
/// the row, so its first row met moments with no row in them, read `x` as
/// it was, and sized its prior on it; each feature's prior now waits for its
/// feature's scale, no row's raw `x` reaches the state, and it keeps its
/// bits too (it failed here on `3d0894b`).
#[test]
fn features_scaled_by_a_power_of_two_predict_the_same_to_the_bit() {
    for (name, build, weighted) in cases() {
        if name.contains("simplex") {
            continue;
        }
        let run = |c: f64| {
            let mut m = build();
            let mut s = 13u64;
            let mut preds = Vec::new();
            for i in 0..300 {
                let (x, y, w) = row(&mut s, i, weighted, c);
                let p = m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, w);
                preds.extend(p.iter().map(|p| p.to_bits()));
            }
            (preds, m.coef())
        };
        let (one, coef) = run(1.0);
        let (scaled, coef_scaled) = run(128.0);
        assert_eq!(scaled, one, "{name}");
        let off = usize::from(!name.contains("origin"));
        for (a, b) in coef_scaled.iter().zip(&coef) {
            for i in 0..a.len() {
                let want = if i < off { b[i] } else { b[i] / 128.0 };
                assert_eq!(
                    a[i].to_bits(),
                    want.to_bits(),
                    "{name}, slot {i}: {a:?} against {b:?}"
                );
            }
        }
    }
}

/// A state saved one row before the switch, on it, one row after it and
/// well past it, written to bytes and read back, goes on as the model that
/// never stopped: every prediction and every coefficient to the bit.
#[test]
fn a_save_around_the_switch_goes_on_to_the_bit() {
    for (name, build, weighted) in cases() {
        let last = switch_row(weighted);
        for cut in [last - 1, last, last + 1, last + 20] {
            let mut s = 17u64;
            let rows: Vec<_> = (0..cut + 60)
                .map(|i| row(&mut s, i, weighted, 1.0))
                .collect();
            let mut first = build();
            for (i, (x, y, w)) in rows.iter().enumerate().take(cut + 1) {
                first.step(x, y, if i == 0 { 0.0 } else { 1.0 }, *w);
            }
            let mut back = first.load(&first.save());
            let mut whole = build();
            for (i, (x, y, w)) in rows.iter().enumerate() {
                let d = if i == 0 { 0.0 } else { 1.0 };
                let want = whole.step(x, y, d, *w);
                if i > cut {
                    let got = back.step(x, y, d, *w);
                    let bits = |v: &[f64]| v.iter().map(|p| p.to_bits()).collect::<Vec<_>>();
                    assert_eq!(
                        bits(&got),
                        bits(&want),
                        "{name}, saved after row {cut}: row {i}"
                    );
                    assert_eq!(
                        back.coef(),
                        whole.coef(),
                        "{name}, saved after row {cut}: row {i}"
                    );
                }
            }
        }
    }
}
