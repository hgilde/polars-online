//! The warm-up refit prototype (`RobustCfg::warm_rows`, branch
//! `huber-warmup`; docs/HUBER.md §3): the replay rebuilds the sums the
//! stream built, the refit fires on a row count, holds through a save and a
//! load mid-buffer, and ignores zero-weight rows.

use super::*;
use crate::OnlineModel;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn base(delta: f64, decay: Decay, warm_rows: u32) -> RobustCfg {
    RobustCfg {
        n_features: 2,
        n_targets: 2,
        fit_intercept: true,
        decay,
        loss: RobustLoss::Huber { delta },
        ridge: 1e-6,
        standardize: false,
        min_weight: 3.0,
        solve_every: 0.0,
        max_rows_between_solves: u32::MAX,
        solve_share: None,
        quantile_eps: 1e-3,
        huber_scale: HuberScale::Mad,
        huber_scale_decay: None,
        feature_sets: Vec::new(),
        ridge_scale: false,
        emit_scale: false,
        warm_rows,
    }
}

/// One row: the features, the two targets and the weight.
type Row = ([f64; 2], [Option<f64>; 2], f64);

/// A two-target stream at a level of 1,000 with one-sided outliers, a null
/// target every 17 rows and, where asked, a zero-weight row every 9.
fn rows(n: usize, zeros: bool) -> Vec<Row> {
    let mut s = 5u64;
    (0..n)
        .map(|i| {
            let x = [lcg(&mut s), lcg(&mut s)];
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

/// At `huber_delta = inf` every Huber weight is 1, so the refit's replay
/// must rebuild the sums the stream built, and with a solve on every row
/// nothing downstream can tell: every prediction to the bit, decayed or not.
#[test]
fn the_replay_rebuilds_the_streams_sums_to_the_bit() {
    for decay in [Decay::Lam(1.0), Decay::Lam(0.995)] {
        let data = rows(400, true);
        let plain = run(
            &mut Robust::new(base(f64::INFINITY, decay, 0)).unwrap(),
            &data,
        );
        let warm = run(
            &mut Robust::new(base(f64::INFINITY, decay, 60)).unwrap(),
            &data,
        );
        assert_eq!(bits(&plain), bits(&warm), "{decay:?}");
    }
}

/// The refit moves the fit where the Huber cut binds: after it the
/// predictions part from the plain stream's, and not before.
#[test]
fn the_refit_fires_on_the_row_the_count_is_reached() {
    let data = rows(300, false);
    let plain = run(
        &mut Robust::new(base(1.345, Decay::Lam(1.0), 0)).unwrap(),
        &data,
    );
    let mut m = Robust::new(base(1.345, Decay::Lam(1.0), 100)).unwrap();
    let warm = run(&mut m, &data);
    // Target 0 is null on rows 3, 20, 37, 54, 71, 88, 105, ...: its 100th
    // learned row is row 106 (seven nulls before it), so row 107 is the
    // first its refit can move.
    let first = (0..300).find(|&i| plain[i][0].to_bits() != warm[i][0].to_bits());
    assert_eq!(first, Some(107));
    assert!(m.warm.iter().all(|b| b.done && b.rows.is_empty()));
}

/// Saved and loaded mid-buffer, the model goes on as one that never
/// stopped, to the bit: the buffer is state.
#[test]
fn a_save_and_load_mid_buffer_goes_on_to_the_bit() {
    let data = rows(300, true);
    let cfg = base(1.345, Decay::Lam(0.999), 120);
    let whole = run(&mut Robust::new(cfg.clone()).unwrap(), &data);
    let mut m = Robust::new(cfg).unwrap();
    let mut parts = run(&mut m, &data[..70]);
    assert!(m.warm.iter().all(|b| !b.done && !b.rows.is_empty()));
    let bytes = rmp_serde::to_vec(&m.state()).unwrap();
    let mut back = Robust::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    parts.extend(run(&mut back, &data[70..]));
    assert_eq!(bits(&whole), bits(&parts));
}

/// A zero-weight row is clock alone (hard rule 9): it is not buffered and
/// does not count, so without decay a stream with such rows predicts every
/// other row as the stream without them does, to the bit -- and a first
/// row of weight 0 poisons nothing.
#[test]
fn zero_weight_rows_neither_count_nor_move_the_refit() {
    let with = rows(300, true);
    let without: Vec<_> = with.iter().filter(|r| r.2 > 0.0).cloned().collect();
    let cfg = base(1.345, Decay::Lam(1.0), 80);
    let mut lead = vec![([0.3, -0.2], [Some(1000.0), Some(0.0)], 0.0)];
    lead.extend(with.iter().cloned());
    let a = run(&mut Robust::new(cfg.clone()).unwrap(), &lead);
    let b = run(&mut Robust::new(cfg).unwrap(), &without);
    let kept: Vec<Vec<f64>> = lead
        .iter()
        .zip(a)
        .filter(|(r, _)| r.2 > 0.0)
        .map(|(_, p)| p)
        .collect();
    assert_eq!(bits(&kept), bits(&b));
}

/// The setting is Huber's alone, and not beside the sum-scale prior.
#[test]
fn the_warm_up_is_refused_off_huber() {
    let mut q = base(1.345, Decay::Lam(1.0), 10);
    q.loss = RobustLoss::Quantile { tau: 0.5 };
    q.huber_scale = HuberScale::Rms;
    assert!(Robust::new(q).is_err());
    let mut s = base(1.345, Decay::Lam(1.0), 10);
    s.ridge_scale = true;
    assert!(Robust::new(s).is_err());
}
