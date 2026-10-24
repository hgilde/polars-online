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
