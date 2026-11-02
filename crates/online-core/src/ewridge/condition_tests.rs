//! A solve past the condition at which it keeps a digit is withheld and
//! counted (docs/PLAN.md task 278, D4): `ewridge`'s side, where the
//! estimate (`crate::solve::SpdFactor::carries_no_digit`) meets the fit,
//! the readiness statistics and the state.

use super::*;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn cfg(ridge: f64) -> EwRidgeCfg {
    EwRidgeCfg {
        n_features: 3,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(f64::INFINITY),
        ridge: vec![ridge],
        feature_sets: vec![],
        standardize: false,
        ridge_scale: false,
        coef_prior: None,
        session_shrink: None,
        long_half_life: None,
        min_weight: 0.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        gram_block_rows: 0,
        target_gaps: TargetGaps::OwnRows,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

/// `x₁ = x₀ + eps·b`: at `eps = 1e-11` the centred correlation matrix's
/// condition is near `1e22`, past `1/u`, and at `1e-4` near `1e8`.
fn rows(n: usize, eps: f64) -> Vec<([f64; 3], f64)> {
    let mut s = 278u64;
    (0..n)
        .map(|_| {
            let (a, b, c) = (lcg(&mut s), lcg(&mut s), lcg(&mut s));
            let x = [2.0 + a, 2.0 + a + eps * b, -1.0 + c];
            let y = 0.7 + x[0] - 2.0 * x[1] + 0.5 * x[2] + 0.1 * lcg(&mut s);
            (x, y)
        })
        .collect()
}

/// What one row of a stream left: whether the fit is withheld as
/// ill-conditioned, the failures its solve added, and the fit.
struct Seen {
    ill: bool,
    added: u64,
    coef: Vec<f64>,
}

fn stream(model: &mut EwRidge, rows: &[([f64; 3], f64)]) -> Vec<Seen> {
    rows.iter()
        .enumerate()
        .map(|(i, (x, y))| {
            let before = model.solve_failures;
            model.step(x, &[Some(*y)], f64::from(u8::from(i > 0)), 1.0);
            Seen {
                ill: model.ill_conditioned(),
                added: model.solve_failures - before,
                coef: model.coefficients().unwrap()[0].clone(),
            }
        })
        .collect()
}

/// Past `1/u` some factorizations succeed, and each of those is withheld:
/// no coefficients, no data shares, no estimation variance, no prediction,
/// and one failure counted for its solve. The rest needed jitter, and are
/// counted and reported as before. Inside the limit nothing is withheld.
#[test]
fn a_solve_past_the_limit_is_withheld_counted_and_said() {
    let data = rows(800, 1e-11);
    let mut model = EwRidge::new(cfg(0.0)).unwrap();
    let seen = stream(&mut model, &data);
    let (mut ill, mut jittered) = (0, 0);
    for (i, s) in seen.iter().enumerate().skip(10) {
        if s.ill {
            ill += 1;
            assert_eq!(
                s.added, 1,
                "row {i}: an ill-conditioned solve is one failure"
            );
            assert!(s.coef.iter().all(|v| v.is_nan()), "row {i}: {:?}", s.coef);
        } else {
            jittered += usize::from(s.added > 0);
            assert!(
                s.coef.iter().all(|v| v.is_finite()),
                "row {i}: {:?}",
                s.coef
            );
        }
    }
    // Evidence that both paths ran (measured 2026-10-25: of rows 10 to
    // 799, 696 withheld and 94 jittered).
    // Which solves rounding lets through past `1/u` is the platform's
    // arithmetic's, so the counts are held loosely.
    assert!(
        ill > 50 && jittered > 10,
        "withheld {ill}, jittered {jittered}"
    );
    // A withheld fit, as the stream reads it: the model stopped at the last
    // row whose solve was withheld.
    let at = (0..800).rev().find(|&i| seen[i].ill).unwrap();
    let mut model = EwRidge::new(cfg(0.0)).unwrap();
    stream(&mut model, &data[..=at]);
    assert!(model.ill_conditioned());
    let x = [2.0, 2.0, -1.0];
    assert!(model.predict(&x, 1.0).pred[0].is_nan());
    let mut infl = Vec::new();
    assert!(model.error_inflation_into(&mut infl));
    assert!(infl[0].is_nan(), "{infl:?}");
    model.set_keep_factor(true);
    let mut row = Vec::new();
    assert!(model.row_error_inflation_into(&x, 1.0, &mut row));
    assert!(row[0].is_nan(), "{row:?}");
    let support = model.support_coef().unwrap();
    assert!(support[0].iter().all(|v| v.is_nan()), "{support:?}");

    // Inside the limit -- a correlation of `1 - 1e-8`, or the same design
    // under a ridge -- nothing is withheld and nothing counted once the rows
    // outnumber the features. Before that the centred system is singular,
    // and a factorization that rounding lets through is withheld too (row
    // 2 at `1e-4`: three rows, three slopes), as it should be.
    for (eps, ridge) in [(1e-4, 0.0), (1e-11, 1e-6)] {
        let mut model = EwRidge::new(cfg(ridge)).unwrap();
        let seen = stream(&mut model, &rows(800, eps));
        assert!(seen[10..].iter().all(|s| !s.ill), "eps {eps} ridge {ridge}");
        assert!(
            seen[10..].iter().all(|s| s.added == 0),
            "eps {eps} ridge {ridge}"
        );
    }
}

/// A withheld fit is told from a missing one by what the state already
/// holds -- the slot names the system it was solved from -- so a state
/// saved there loads withheld, and goes on as the uninterrupted stream does,
/// to the bit.
#[test]
fn a_withheld_fit_survives_a_state_round_trip() {
    let data = rows(800, 1e-11);
    let mut model = EwRidge::new(cfg(0.0)).unwrap();
    let seen = stream(&mut model, &data[..600]);
    let at = (0..600)
        .rev()
        .find(|&i| seen[i].ill)
        .expect("a withheld row");
    let mut model = EwRidge::new(cfg(0.0)).unwrap();
    stream(&mut model, &data[..=at]);
    assert!(model.ill_conditioned());
    let bytes = rmp_serde::to_vec(&model.state()).unwrap();
    let state: State = rmp_serde::from_slice(&bytes).unwrap();
    let mut restored = EwRidge::restore(&state).unwrap();
    assert!(
        restored.ill_conditioned(),
        "row {at}: the load forgot the withheld fit"
    );
    let a = stream(&mut model, &data[at + 1..]);
    let b = stream(&mut restored, &data[at + 1..]);
    for (i, (p, q)) in a.iter().zip(&b).enumerate() {
        assert_eq!(p.ill, q.ill, "row {}", at + 1 + i);
        assert_eq!(p.added, q.added, "row {}", at + 1 + i);
        assert!(
            p.coef
                .iter()
                .zip(&q.coef)
                .all(|(u, v)| u.to_bits() == v.to_bits()),
            "row {}",
            at + 1 + i
        );
    }
}

/// A slot with no fit for another reason -- no row yet, a target with no
/// weight -- is not called ill-conditioned: before the first solve, and on a
/// target no row has carried.
#[test]
fn a_missing_fit_is_not_a_withheld_one() {
    let mut model = EwRidge::new(cfg(0.0)).unwrap();
    assert!(!model.ill_conditioned());
    let mut c = cfg(0.0);
    c.n_targets = 2;
    let mut two = EwRidge::new(c).unwrap();
    for (i, (x, y)) in rows(50, 1e-2).iter().enumerate() {
        let d = f64::from(u8::from(i > 0));
        model.step(x, &[Some(*y)], d, 1.0);
        two.step(x, &[Some(*y), None], d, 1.0);
    }
    assert!(!model.ill_conditioned());
    assert!(two.coefficients().unwrap()[1].iter().all(|v| v.is_nan()));
    assert!(!two.ill_conditioned());
}

/// A withheld system is one `(Gram, combo)` of several, and only its slots
/// are withheld: two targets, the second absent from every third row, so
/// each has a Gram of its own (`own_rows`), under a grid of two ridges, the
/// unpenalized one second. Every slot of the penalized combo keeps its fit
/// and its kept system (its leverage is a number), every withheld slot is
/// the unpenalized combo's, and the withheld fits of both Grams are seen.
#[test]
fn only_the_withheld_systems_slots_are_withheld() {
    let mut c = cfg(0.0);
    c.n_targets = 2;
    c.ridge = vec![1e-6, 0.0];
    let mut model = EwRidge::new(c).unwrap();
    model.set_keep_factor(true);
    let (nc, x0) = (2, [2.0, 2.0, -1.0]);
    let mut withheld = [0usize; 2];
    let mut row = Vec::new();
    for (i, (x, y)) in rows(400, 1e-11).iter().enumerate() {
        let second = (i % 3 != 0).then_some(-*y);
        model.step(x, &[Some(*y), second], f64::from(u8::from(i > 0)), 1.0);
        if i < 20 {
            continue;
        }
        let coef = model.coefficients().unwrap();
        assert!(model.row_error_inflation_into(&x0, 1.0, &mut row));
        for (j, count) in withheld.iter_mut().enumerate() {
            let (penalized, free) = (j * nc, j * nc + 1);
            assert!(
                coef[penalized].iter().all(|v| v.is_finite()),
                "row {i} target {j}"
            );
            assert!(row[penalized].is_finite(), "row {i} target {j}: {row:?}");
            if coef[free].iter().all(|v| v.is_nan()) {
                *count += 1;
                assert!(row[free].is_nan(), "row {i} target {j}: {row:?}");
            } else {
                assert!(
                    coef[free].iter().all(|v| v.is_finite()),
                    "row {i} target {j}"
                );
                assert!(row[free].is_finite(), "row {i} target {j}: {row:?}");
            }
        }
        assert_eq!(
            model.ill_conditioned(),
            (0..2).any(|j| coef[j * nc + 1].iter().all(|v| v.is_nan())),
            "row {i}"
        );
    }
    assert!(withheld[0] > 20 && withheld[1] > 20, "{withheld:?}");
}

/// A configuration of `k` features for the ridge's shortcut (task 306): a
/// ridge of `1e-6` (the default), a half-life of 200 rows, a solve every
/// row.
fn wide(k: usize) -> EwRidgeCfg {
    EwRidgeCfg {
        n_features: k,
        ridge: vec![1e-6],
        decay: Decay::Halflife(200.0),
        ..cfg(1e-6)
    }
}

/// What a stream left on every row, to the bit: predictions, the fit, the
/// withheld flag and the failures; and how many solves the ridge's
/// shortcut settled on the way.
fn bits(cfg: &EwRidgeCfg, rows: usize, scale: f64, off: bool) -> (Vec<u64>, u64) {
    use crate::solve::condition_switches::{SHORTCUT_OFF, SHORTCUTS};
    let k = cfg.n_features;
    let mut model = EwRidge::new(cfg.clone()).unwrap();
    let mut s = 306u64 + k as u64;
    SHORTCUT_OFF.with(|c| c.set(off));
    let before = SHORTCUTS.with(std::cell::Cell::get);
    let mut out = Vec::new();
    for i in 0..rows {
        let x: Vec<f64> = (0..k).map(|_| scale * (1.0 + lcg(&mut s))).collect();
        let y = 0.3
            + x.iter()
                .enumerate()
                .map(|(j, v)| v / (j + 1) as f64)
                .sum::<f64>()
            + 0.1 * lcg(&mut s);
        let p = model.predict(&x, 1.0);
        out.extend(p.pred.iter().map(|v| v.to_bits()));
        model.step(&x, &[Some(y)], f64::from(u8::from(i > 0)), 1.0);
        out.push(u64::from(model.ill_conditioned()));
        out.push(model.solve_failures);
        if let Some(c) = model.coefficients() {
            out.extend(c.iter().flatten().map(|v| v.to_bits()));
        }
    }
    let settled = SHORTCUTS.with(std::cell::Cell::get) - before;
    SHORTCUT_OFF.with(|c| c.set(false));
    (out, settled)
}

/// The ridge's shortcut fires on a live Gram's centred solves and changes
/// nothing: forty features, a solve every row, unscaled and standardized,
/// read every row to the bit with the shortcut and with it off (the long
/// way on every solve), and settling nearly every solve; under a window,
/// the solves before the first row leaves it; on a blocked Gram (a solve
/// every fourth row, blocks of eight), every one of its thirty solves (all but the first,
/// standardized, which has no system yet), by
/// the merges' kept bound (task 308). Where it must not run -- a window's
/// truncated Gram, `session_shrink`'s twin, the raw normal equations of
/// `ridge_scale` and of a fit through the origin -- it settles none, and
/// where the ridge does not hold the system inside the limit (features of
/// spread 1,000 against a ridge of `1e-6`), none past the first row: the
/// estimate decides those, as before.
#[test]
fn the_ridge_settles_a_live_grams_solves_to_the_same_bits() {
    let k = 40;
    let rows = 120;
    let base = wide(k);
    // How many of the solves each case settles, at least and at most.
    let all = (rows - 5, rows);
    let none = (0, 0);
    let cases: Vec<(&str, EwRidgeCfg, f64, (usize, usize))> = vec![
        ("default", base.clone(), 1.0, all),
        (
            "standardized",
            EwRidgeCfg {
                standardize: true,
                ..base.clone()
            },
            1.0,
            all,
        ),
        // Until a row leaves the window the fit reads the live Gram, and
        // the shortcut settles those solves (the first 60 rows); from then
        // on the window's truncation, none.
        (
            "window",
            EwRidgeCfg {
                window: Some(60.0),
                ..base.clone()
            },
            1.0,
            (50, 61),
        ),
        (
            "blocked",
            EwRidgeCfg {
                gram_block_rows: 8,
                solve_every: 4.0,
                max_rows_between_solves: 4,
                ..base.clone()
            },
            1.0,
            (30, 30),
        ),
        (
            "blocked, standardized",
            EwRidgeCfg {
                gram_block_rows: 8,
                solve_every: 4.0,
                max_rows_between_solves: 4,
                standardize: true,
                ..base.clone()
            },
            1.0,
            (29, 30),
        ),
        // The blocked Gram's first solve alone, where `C` is still 0.
        (
            "blocked, spread 1,000",
            EwRidgeCfg {
                gram_block_rows: 8,
                solve_every: 4.0,
                max_rows_between_solves: 4,
                ..base.clone()
            },
            1000.0,
            (1, 1),
        ),
        (
            "session_shrink",
            EwRidgeCfg {
                session_shrink: Some(0.5),
                long_half_life: Some(800.0),
                ..base.clone()
            },
            1.0,
            none,
        ),
        (
            "ridge_scale",
            EwRidgeCfg {
                ridge_scale: true,
                ..base.clone()
            },
            1.0,
            none,
        ),
        (
            "through the origin",
            EwRidgeCfg {
                fit_intercept: false,
                ..base.clone()
            },
            1.0,
            none,
        ),
        // The first row's alone, where `C` is still 0 and `A` is the ridge.
        ("spread 1,000", base.clone(), 1000.0, (1, 1)),
    ];
    for (name, cfg, scale, (least, most)) in cases {
        let (on, settled) = bits(&cfg, rows, scale, false);
        let (off, none) = bits(&cfg, rows, scale, true);
        assert_eq!(none, 0, "{name}: switched off");
        assert!(on == off, "{name}: the shortcut moved a bit");
        let settled = settled as usize;
        assert!(
            (least..=most).contains(&settled),
            "{name}: {settled} of {rows} settled"
        );
    }
}

/// What a blocked fit left on every row, to the bit, and how many solves
/// the ridge's shortcut settled: `k` features of spread `scale`, the first
/// two equal to within `1e-11`, and with `shifted` every block opened by a
/// row of weight `1e-46` at `1e20` -- the row its mean is summed about, so
/// that each merge subtracts a residue of about `1e5` against a spread of 1
/// (task 308).
fn blocked_bits(
    cfg: &EwRidgeCfg,
    rows: usize,
    scale: f64,
    shifted: bool,
    off: bool,
) -> (Vec<u64>, u64) {
    use crate::solve::condition_switches::{SHORTCUT_OFF, SHORTCUTS};
    let k = cfg.n_features;
    let block = cfg.gram_block_rows;
    let mut model = EwRidge::new(cfg.clone()).unwrap();
    let mut s = 308u64 + k as u64;
    SHORTCUT_OFF.with(|c| c.set(off));
    let before = SHORTCUTS.with(std::cell::Cell::get);
    let mut out = Vec::new();
    for i in 0..rows {
        let mut x: Vec<f64> = (0..k).map(|_| scale * 3f64.sqrt() * lcg(&mut s)).collect();
        x[1] = x[0] + 1e-11 * lcg(&mut s);
        let mut w = 1.0;
        if shifted && i % block == 0 {
            x.iter_mut()
                .for_each(|v| *v = 1e20 * (1.0 + 0.01 * lcg(&mut s)));
            x[1] = x[0];
            w = 1e-46;
        }
        let y = 0.3 + x[0] - 0.5 * x[2] + 0.1 * lcg(&mut s);
        let p = model.predict(&x, 1.0);
        out.extend(p.pred.iter().map(|v| v.to_bits()));
        model.step(&x, &[Some(y)], 1.0, w);
        out.push(u64::from(model.ill_conditioned()));
        out.push(model.solve_failures);
        if let Some(c) = model.coefficients() {
            out.extend(c.iter().flatten().map(|v| v.to_bits()));
        }
    }
    let settled = SHORTCUTS.with(std::cell::Cell::get) - before;
    SHORTCUT_OFF.with(|c| c.set(false));
    (out, settled)
}

/// A blocked fit of `k` features: blocks of `block` rows, a solve every
/// `block` rows, a half-life of `half_life` rows and a ridge of `1e-6`.
fn blocked(k: usize, block: usize, half_life: f64, standardize: bool) -> EwRidgeCfg {
    EwRidgeCfg {
        n_features: k,
        ridge: vec![1e-6],
        decay: Decay::Halflife(half_life),
        gram_block_rows: block,
        solve_every: block as f64,
        max_rows_between_solves: block as u32,
        standardize,
        ..cfg(1e-6)
    }
}

/// On a blocked Gram the shortcut changes nothing, on the streams built to
/// break a bound read from the matrix too: twelve features, the first two
/// equal to within `1e-11`, blocks of sixteen, with and without each block
/// opened by a row of weight `1e-46` at `1e20`, unscaled and standardized,
/// read every row to the bit with the shortcut on and off. It settles
/// solves on every one of them: the kept bound reads the residues the
/// shifted rows leave, about `22 ε ω h²` a merge, and the ridge still
/// outweighs it.
#[test]
fn a_blocked_grams_shortcut_moves_no_bit() {
    for shifted in [false, true] {
        for standardize in [false, true] {
            let cfg = blocked(12, 16, 200.0, standardize);
            let (on, settled) = blocked_bits(&cfg, 480, 1.0, shifted, false);
            let (off, none) = blocked_bits(&cfg, 480, 1.0, shifted, true);
            let what = format!("shifted {shifted} std {standardize}");
            assert_eq!(none, 0, "{what}: switched off");
            assert!(on == off, "{what}: the shortcut moved a bit");
            eprintln!("{what}: settled {settled} of 30");
            assert!(settled >= 25, "{what}: settled {settled} of 30");
        }
    }
}

/// Activation on G4's shape: two hundred unit-variance features, blocks of
/// 64 and a solve every 64 rows, a half-life of 1,000 rows and the default
/// ridge of `1e-6`, over one group's 2,000 rows. The ridge settles every
/// solve once the first block has merged; at a spread of 100 (variances
/// near `1e4`) it settles none past the first, where `C` is still 0.
#[test]
fn a_blocked_fit_of_two_hundred_features_settles_by_the_ridge() {
    let cfg = blocked(200, 64, 1000.0, false);
    for (scale, least, most) in [(1.0, 30, 32), (100.0, 0, 1)] {
        let (_, settled) = blocked_bits(&cfg, 2000, scale, false, false);
        eprintln!("scale {scale}: settled {settled}");
        assert!(
            (least..=most).contains(&settled),
            "scale {scale}: {settled}"
        );
    }
}
