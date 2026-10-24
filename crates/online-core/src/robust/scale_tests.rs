//! The robust scale a Huber cut is drawn in (docs/PLAN.md task 271,
//! docs/HUBER.md): the plain EW root mean square, `huber_scale = "rms"`,
//! held to the bits it had before the MAD was added, and the quantile
//! fit's band, which shares the code, held to the same.

use super::*;
use crate::OnlineModel;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn base(loss: RobustLoss) -> RobustCfg {
    RobustCfg {
        n_features: 2,
        n_targets: 2,
        fit_intercept: true,
        decay: Decay::Lam(0.9914),
        loss,
        ridge: 1e-6,
        standardize: false,
        min_weight: 3.0,
        solve_every: 0.0,
        max_rows_between_solves: 4,
        solve_share: None,
        quantile_eps: 0.2,
        huber_scale: HuberScale::Rms,
        huber_scale_decay: None,
        feature_sets: Vec::new(),
        ridge_scale: false,
    }
}

/// Every prediction of a 600-row, two-target stream digested (FNV-1a over
/// the bits, NaN folded to one pattern), and three picks: under a literal
/// decay at unit steps, so no libm result is on the path and the bits are
/// every platform's (`sgd.rs`'s `the_huber_and_squared_losses_did_not_move`
/// says why). Irregular weights, a zero weight every 23 rows, a null target
/// every 17, and outliers the cut binds on.
fn digest(cfg: RobustCfg) -> (u64, Vec<f64>) {
    let mut m = Robust::new(cfg).unwrap();
    let mut s = 17u64;
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut got = Vec::new();
    for i in 0..600 {
        let x = [lcg(&mut s), 3.0 + lcg(&mut s)];
        let outlier = if i % 41 == 7 { 25.0 } else { 0.0 };
        let y0 = (i % 17 != 3).then(|| 5.0 + 2.0 * x[0] - x[1] + 0.3 * lcg(&mut s) + outlier);
        let y1 = Some(1000.0 - x[0] + 0.5 * lcg(&mut s) - outlier);
        let d = if i == 0 { 0.0 } else { 1.0 };
        let w = if i % 23 == 9 {
            0.0
        } else {
            0.5 + lcg(&mut s).abs()
        };
        let out = m.step(&x, &[y0, y1], d, w);
        for p in out.pred.iter().copied().chain([out.n_eff]) {
            let b = if p.is_nan() {
                0x7ff8_0000_0000_0000
            } else {
                p.to_bits()
            };
            for byte in b.to_le_bytes() {
                h = (h ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
            }
        }
        if matches!(i, 50 | 300 | 599) {
            got.extend(out.pred.iter().copied());
        }
    }
    (h, got)
}

/// `huber_scale = "rms"` is the scale `huber` drew its cut in before the
/// MAD was added, to the bit, and so is the quantile band, which reads the
/// same variance: the digests and picks were taken on the build before task
/// 271 (`b418fc3`).
#[test]
fn the_rms_scale_and_the_quantile_band_did_not_move() {
    let huber = digest(base(RobustLoss::Huber { delta: 1.345 }));
    let quantile = digest(base(RobustLoss::Quantile { tau: 0.7 }));
    println!("huber {huber:x?}\nquantile {quantile:x?}");
    for (what, (h, got), (digest, picks)) in [
        (
            "huber",
            huber,
            (
                0x6f2e_cdea_ece4_e74a_u64,
                [
                    2.4384695906632388,
                    1000.0104319094211,
                    4.025336608447544,
                    999.4020032545193,
                    1.590112844518908,
                    1000.4997202341584,
                ],
            ),
        ),
        (
            "quantile",
            quantile,
            (
                0x4c68_f651_7faa_dfde_u64,
                [
                    4.073600568098862,
                    1000.7795194598039,
                    4.4341509298869415,
                    1000.0355330770761,
                    1.9325823542509095,
                    1000.8926474200136,
                ],
            ),
        ),
    ] {
        assert_eq!(h, digest, "{what}: picks {got:?}");
        for (g, w) in got.iter().zip(picks) {
            assert_eq!(g.to_bits(), f64::to_bits(w), "{what}: {got:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// The MAD (docs/PLAN.md task 271, docs/HUBER.md §2-§4).
// ---------------------------------------------------------------------------

fn huber_cfg(scale: HuberScale) -> RobustCfg {
    RobustCfg {
        n_features: 1,
        n_targets: 1,
        decay: Decay::Halflife(f64::INFINITY),
        max_rows_between_solves: 1,
        huber_scale: scale,
        ..base(RobustLoss::Huber { delta: 1.345 })
    }
}

fn sgd_cfg(loss: crate::SgdLoss, scale: HuberScale) -> crate::SgdCfg {
    crate::SgdCfg {
        n_features: 1,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(f64::INFINITY),
        loss,
        learning_rate: 0.01,
        schedule: crate::LearningRate::Constant,
        l2: 0.0,
        min_weight: 0.0,
        standardize: true,
        clip_gradient: 1e3,
        constraint: None,
        strict_binary: false,
        huber_scale: scale,
        huber_scale_decay: None,
    }
}

/// HUBER.md §2's stream: `y = 1000 + 2x + 0.5ε`, an outlier of +50 on a
/// share of the rows with `x > 0`, which tilts the slope; `x` and `ε`
/// uniform on `[-√3, √3]`, unit variance, so no libm is on the path.
fn contaminated(n: usize, frac: f64, seed: u64) -> Vec<(f64, f64)> {
    let mut s = seed;
    let r3 = 3f64.sqrt();
    (0..n)
        .map(|_| {
            let x = r3 * lcg(&mut s);
            let e = r3 * lcg(&mut s);
            let u = 0.5 * (lcg(&mut s) + 1.0);
            let lift = if x > 0.0 && u < frac { 50.0 } else { 0.0 };
            (x, 1000.0 + 2.0 * x + 0.5 * e + lift)
        })
        .collect()
}

fn huber_slope(scale: HuberScale, rows: &[(f64, f64)]) -> f64 {
    let mut m = Robust::new(huber_cfg(scale)).unwrap();
    for (i, &(x, y)) in rows.iter().enumerate() {
        m.step(&[x], &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    m.coefficients().unwrap()[0][1]
}

fn sgd_slope(loss: crate::SgdLoss, scale: HuberScale, rows: &[(f64, f64)]) -> f64 {
    let mut m = crate::Sgd::new(sgd_cfg(loss, scale)).unwrap();
    for (i, &(x, y)) in rows.iter().enumerate() {
        m.step(&[x], &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    m.coefficients()[0][1]
}

/// HUBER.md §2 and §3 at 1% and 5% of the rows (2% and 10% of those with
/// `x > 0`), through the model: the MAD holds the slope near the truth
/// where the RMS, inflated by the start-up residuals at the level's size
/// and by the outliers it is meant to catch, gives back most of least
/// squares' error. Measured on this stream: MAD 2.004 / 2.035, RMS 2.05 /
/// 2.67 (statsmodels' converged `RLM` with the MAD lands at 2.008 / 2.039
/// on the normal stream of `tests/test_huber_scale.py`).
#[test]
fn the_mad_scale_holds_the_slope_where_the_rms_breaks() {
    for (frac, mad_tol, rms_off) in [(0.02, 0.02, 0.03), (0.10, 0.06, 0.4)] {
        let rows = contaminated(20_000, frac, 11);
        let mad = huber_slope(HuberScale::Mad, &rows);
        let rms = huber_slope(HuberScale::Rms, &rows);
        println!("frac {frac}: mad {mad} rms {rms}");
        assert!((mad - 2.0).abs() < mad_tol, "frac {frac}: mad {mad}");
        assert!((rms - 2.0).abs() > rms_off, "frac {frac}: rms {rms}");
    }
}

/// C-1 (review round 9): `sgd(loss="huber")` on a target at a level of
/// 1,000 with no decay was squared loss to the bit under the RMS -- the
/// first residuals, the level's size, set `s` for good, so no row was ever
/// clipped -- and under the MAD its slope is nearer the truth than squared
/// loss's.
#[test]
fn sgd_huber_at_a_level_is_not_squared_loss_under_the_mad() {
    let rows = contaminated(20_000, 0.10, 12);
    let squared = sgd_slope(crate::SgdLoss::Squared, HuberScale::Rms, &rows);
    let huber = crate::SgdLoss::Huber { delta: 1.345 };
    let rms = sgd_slope(huber, HuberScale::Rms, &rows);
    let mad = sgd_slope(huber, HuberScale::Mad, &rows);
    println!("squared {squared} rms {rms} mad {mad}");
    assert_eq!(rms.to_bits(), squared.to_bits(), "C-1 under the RMS, kept");
    assert!(
        (mad - 2.0).abs() < 0.25 * (squared - 2.0).abs(),
        "mad {mad} squared {squared}"
    );
}

/// The MAD scale is the EW median of `|e|` over `Φ⁻¹(3/4)` within the
/// sketch's `tanh(1/128)`: checked at every row of a decayed stream with
/// irregular weights and zero weights against the exact weighted median of
/// the residuals kept with their decayed weights (the smallest `|e|` whose
/// cumulative weight reaches half the total). The residuals are recomputed
/// here from the model's own predictions, so the check reads the sketch the
/// cut was drawn in, not a copy of its arithmetic.
#[test]
fn the_mad_scale_is_the_ew_median_within_the_sketchs_accuracy() {
    let mut cfg = huber_cfg(HuberScale::Mad);
    cfg.decay = Decay::Lam(0.99);
    cfg.huber_scale_decay = Some(Decay::Lam(0.97));
    let mut m = Robust::new(cfg).unwrap();
    let rows = contaminated(3000, 0.10, 13);
    let mut kept: Vec<(f64, f64)> = Vec::new();
    let mut s = 99u64;
    let mut checked = 0;
    for (i, &(x, y)) in rows.iter().enumerate() {
        let d = if i == 0 { 0.0 } else { 1.0 };
        let w = if i % 9 == 4 {
            0.0
        } else {
            0.5 + lcg(&mut s).abs()
        };
        // The scale the row's cut is drawn in, before the row.
        if let Some(got) = m.residual_scale(0) {
            let total: f64 = kept.iter().map(|k| k.1).sum();
            let mut sorted = kept.clone();
            sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut acc = 0.0;
            let median = sorted
                .iter()
                .find(|(_, w)| {
                    acc += w;
                    acc >= 0.5 * total
                })
                .unwrap()
                .0;
            let want = median / NORMAL_Q3;
            assert!(
                (got - want).abs() <= crate::EW_QUANTILE_ALPHA * want * (1.0 + 1e-9),
                "row {i}: {got} against {want}"
            );
            checked += 1;
        }
        let pred = m.step(&[x], &[Some(y)], d, w).pred[0];
        for k in kept.iter_mut() {
            k.1 *= 0.97;
        }
        if pred.is_finite() && w > 0.0 {
            kept.push(((y - pred).abs(), w));
        }
    }
    assert!(checked > 2900, "{checked}");
}

/// A zero-weight row ages the sketch on the scale's clock and adds nothing
/// (hard rule 9): the sketch it leaves is the one a sketch aged alone
/// leaves, byte for byte.
#[test]
fn a_zero_weight_row_ages_the_sketch_and_adds_nothing() {
    let mut cfg = huber_cfg(HuberScale::Mad);
    cfg.huber_scale_decay = Some(Decay::Lam(0.9));
    let mut m = Robust::new(cfg).unwrap();
    for &(x, y) in &contaminated(50, 0.1, 14) {
        m.step(&[x], &[Some(y)], 1.0, 1.0);
    }
    let mut want = m.mad_sketch()[0].clone();
    want.age(0.9);
    m.step(&[0.3], &[Some(1e6)], 1.0, 0.0);
    assert_eq!(m.mad_sketch()[0], want);
}

/// The scale's own memory left at the model's is the model's, to the bit:
/// `huber_scale_decay = Some(decay)` and `None` give one stream. And a
/// shorter one moves it.
#[test]
fn the_scales_memory_defaults_to_the_models() {
    for scale in [HuberScale::Mad, HuberScale::Rms] {
        let mut own = huber_cfg(scale);
        own.decay = Decay::Lam(0.995);
        let mut same = own.clone();
        same.huber_scale_decay = Some(own.decay);
        let mut short = own.clone();
        short.huber_scale_decay = Some(Decay::Lam(0.9));
        let run = |c: RobustCfg| {
            let mut m = Robust::new(c).unwrap();
            contaminated(800, 0.1, 15)
                .iter()
                .map(|&(x, y)| m.step(&[x], &[Some(y)], 1.0, 1.0).pred[0].to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(run(own.clone()), run(same), "{scale:?}");
        assert_ne!(run(own), run(short), "{scale:?}");
    }
}

/// What each configuration refuses: a scale of Huber's on a quantile fit
/// or on an `sgd` loss other than Huber's, and a scale memory a decay
/// refuses.
#[test]
fn the_scale_is_huber_s_alone() {
    let mut q = base(RobustLoss::Quantile { tau: 0.5 });
    q.huber_scale = HuberScale::Mad;
    assert!(Robust::new(q.clone()).unwrap_err().contains("huber_scale"));
    q.huber_scale = HuberScale::Rms;
    q.huber_scale_decay = Some(Decay::Lam(0.9));
    assert!(
        Robust::new(q)
            .unwrap_err()
            .contains("huber_scale_half_life")
    );
    let mut h = huber_cfg(HuberScale::Mad);
    h.huber_scale_decay = Some(Decay::Halflife(0.0));
    assert!(
        Robust::new(h)
            .unwrap_err()
            .contains("huber_scale_half_life")
    );
    let sq = sgd_cfg(crate::SgdLoss::Squared, HuberScale::Mad);
    assert!(crate::Sgd::new(sq).unwrap_err().contains("huber_scale"));
    let mut sq = sgd_cfg(crate::SgdLoss::Squared, HuberScale::Rms);
    sq.huber_scale_decay = Some(Decay::Lam(0.5));
    assert!(crate::Sgd::new(sq).unwrap_err().contains("huber_scale"));
    let mut sh = sgd_cfg(crate::SgdLoss::Huber { delta: 1.0 }, HuberScale::Mad);
    sh.huber_scale_decay = Some(Decay::Lam(1.5));
    assert!(
        crate::Sgd::new(sh)
            .unwrap_err()
            .contains("huber_scale_half_life")
    );
}

/// A state whose sketches are not its scale's is refused: one per target
/// at 0.5 under the MAD, none under the RMS.
#[test]
fn a_state_whose_sketches_are_not_its_scales_is_refused() {
    let mad = Robust::new(huber_cfg(HuberScale::Mad)).unwrap();
    let rms = Robust::new(huber_cfg(HuberScale::Rms)).unwrap();
    let mut lost = mad.clone();
    lost.mad.clear();
    let mut stray = rms.clone();
    stray.mad = mad.mad.clone();
    let mut other = mad.clone();
    other.mad = vec![EwQuantile::new(&[0.25]).unwrap()];
    for (what, m) in [("lost", lost), ("stray", stray), ("level", other)] {
        let err = Robust::restore(&m.state()).unwrap_err().to_string();
        assert!(err.contains("residual sketches"), "{what}: {err}");
    }
    for m in [mad, rms] {
        assert!(Robust::restore(&m.state()).is_ok());
    }
    let sg = crate::Sgd::new(sgd_cfg(
        crate::SgdLoss::Huber { delta: 1.0 },
        HuberScale::Mad,
    ))
    .unwrap();
    let mut v = serde_json::to_value(sg.state()).unwrap();
    v["model"]["Sgd"]["mad"] = serde_json::json!([]);
    // `sgd` checks its layout as it decodes (`SgdV3`).
    let err = serde_json::from_value::<State>(v).unwrap_err().to_string();
    assert!(err.contains("residual sketches"), "{err}");
}

/// The subtle part of schema 63's loader (CLAUDE.md hard rule 5): a state
/// written before it has no `huber_scale` and was fitted under the RMS, so
/// it loads as the RMS with no sketch, where a new configuration's spec
/// default is the MAD (`online_polars`). A `huber` and an `sgd` state with
/// both fields and the sketches taken out, as a 62 build wrote them, load
/// as `"rms"`.
#[test]
fn a_state_without_huber_scale_loads_as_the_rms() {
    let mut h = Robust::new(huber_cfg(HuberScale::Rms)).unwrap();
    let mut sg = crate::Sgd::new(sgd_cfg(
        crate::SgdLoss::Huber { delta: 1.0 },
        HuberScale::Rms,
    ))
    .unwrap();
    for &(x, y) in &contaminated(30, 0.1, 16) {
        h.step(&[x], &[Some(y)], 1.0, 1.0);
        sg.step(&[x], &[Some(y)], 1.0, 1.0);
    }
    for (variant, state) in [("Robust", h.state()), ("Sgd", sg.state())] {
        let mut v = serde_json::to_value(&state).unwrap();
        let model = v["model"][variant].as_object_mut().unwrap();
        model.remove("mad").unwrap();
        let cfg = model["cfg"].as_object_mut().unwrap();
        cfg.remove("huber_scale").unwrap();
        cfg.remove("huber_scale_decay").unwrap();
        let old: State = serde_json::from_value(v).unwrap();
        let back = match variant {
            "Robust" => {
                let m = Robust::restore(&old).unwrap();
                assert_eq!(m.cfg().huber_scale, HuberScale::Rms);
                assert_eq!(m.cfg().huber_scale_decay, None);
                assert!(m.mad_sketch().is_empty());
                m.state()
            }
            _ => {
                let m = crate::Sgd::restore(&old).unwrap();
                assert_eq!(m.cfg().huber_scale, HuberScale::Rms);
                assert!(m.mad_sketch().is_empty());
                m.state()
            }
        };
        assert!(
            rmp_serde::to_vec_named(&back).unwrap() == rmp_serde::to_vec_named(&state).unwrap(),
            "{variant}: the old form loads as the state it was"
        );
    }
}
