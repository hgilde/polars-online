//! The pieces of `quantile`'s refit, each against its definition in the
//! module docs (task 281, the mutants of the pushes of f4f6fa5e and
//! c02e0afd): the smoothed check loss, its change and the size of its
//! terms, the majorizer's row, the stream's row, and the damped step.

use super::*;
use crate::huber_refit::RefitBuf;
use crate::{Decay, HuberScale, RobustCfg, RobustLoss};

/// The check loss `ρ_τ(e) = e·(τ − 1{e < 0})`.
fn check(e: f64, tau: f64) -> f64 {
    if e < 0.0 { e * (tau - 1.0) } else { e * tau }
}

/// `(1/2h) ∫_{−h}^{h} ρ_τ(r − u) du`, by the midpoint rule on 200,000
/// panels: the check loss convolved with a uniform kernel, computed from
/// its definition and sharing no arithmetic with `smoothed_check`. The
/// integrand is linear but at one kink, so the rule's error is a panel's
/// worth there, `O(h/n²)`.
fn convolved(r: f64, h: f64, tau: f64) -> f64 {
    let n = 200_000;
    let du = 2.0 * h / f64::from(n);
    let sum: f64 = (0..n)
        .map(|k| check(r - (-h + (f64::from(k) + 0.5) * du), tau))
        .sum();
    sum * du / (2.0 * h)
}

#[test]
fn the_smoothed_check_loss_is_the_check_loss_convolved_with_the_band() {
    for (h, tau) in [(1.0, 0.75), (0.3, 0.5), (2.5, 0.1)] {
        for r in [-3.1, -1.0, -0.4, -0.05, 0.0, 0.05, 0.4, 1.0, 3.1] {
            let r = r * h;
            let (got, want) = (smoothed_check(r, h, tau), convolved(r, h, tau));
            assert!(
                (got - want).abs() <= 1e-9 * (1.0 + want.abs()),
                "h {h}, tau {tau}, r {r}: {got} against {want}"
            );
        }
    }
}

#[test]
fn psi_is_the_check_losss_slope() {
    for tau in [0.1, 0.5, 0.9] {
        assert_eq!(psi(2.0, tau), tau);
        assert_eq!(psi(-2.0, tau), tau - 1.0);
        assert_eq!(psi(0.0, tau), tau - 1.0, "1{{r <= 0}} at 0");
    }
}

/// `check_change(r, d, h, τ)` is `ρ_h(r − d) − ρ_h(r)` to the rounding of
/// its terms, and the size it gives is the size of the terms its closed form
/// sums: inside the band on both sides `|d|·(|r + r'|/4h + |τ − ½|)`, above
/// it `|d·τ|`, below it `|d·(τ − 1)|`, and across pieces `|ρ_h(r')| +
/// |ρ_h(r)|`, the band's edge `|e| = h` itself outside it, as `ρ_h` reads it.
#[test]
fn a_rows_change_is_the_difference_of_its_losses_and_its_size_its_terms() {
    let (h, tau) = (1.0, 0.75);
    let cases: [(f64, f64, &str); 9] = [
        (0.5, 0.25, "inside"),
        (-0.3, 0.5, "inside"),
        (2.0, 0.5, "above"),
        (-2.0, 0.5, "below"),
        (0.5, 2.0, "across"),
        (-0.5, -3.0, "across"),
        (3.0, 6.0, "across"),
        (1.0, 0.5, "across"),
        (-1.0, -0.5, "across"),
    ];
    for (r, d, piece) in cases {
        let after = r - d;
        let (value, size) = check_change(r, d, h, tau);
        let (p, q) = (smoothed_check(after, h, tau), smoothed_check(r, h, tau));
        assert!(
            (value - (p - q)).abs() <= 4.0 * f64::EPSILON * (p.abs() + q.abs()),
            "{r}, {d}: {value} against {}",
            p - q
        );
        let want = match piece {
            "inside" => d.abs() * ((r + after).abs() / (4.0 * h) + (tau - 0.5f64).abs()),
            "above" => (d * tau).abs(),
            "below" => (d * (tau - 1.0)).abs(),
            _ => p.abs() + q.abs(),
        };
        assert!(
            (size - want).abs() <= 1e-15 * want,
            "{piece} {r}, {d}: size {size} against {want}"
        );
    }
}

/// One weighted row, as `Rows` reads the refit buffer: features `x`,
/// target `y`, weight `w`.
fn one_row(x: [f64; 2], y: f64, w: f64) -> (Vec<f64>, Vec<f64>) {
    let mut b = RefitBuf::new();
    b.learn(&x, y, w, 1.0, 1.0, 9);
    b.take()
}

/// The majorizer's row is a least-squares row at the row's weight and the
/// target `z'β + 2h·ρ_h'(r)`, `ρ_h'` the slope of the smoothed loss: read
/// here by a central difference of `smoothed_check`, at a residual inside
/// the band and on either side of it.
#[test]
fn the_majorizers_row_steps_by_twice_the_band_times_the_losss_slope() {
    let (h, tau, y, w) = (3.0, 0.75, 10.0, 2.5);
    for r in [0.3, -2.0, 4.5, -5.0] {
        let (buf, ages) = one_row([0.1, 0.2], y, w);
        let rows = Rows::new(&buf, &ages, 2);
        let slope = (smoothed_check(r + 1e-6, h, tau) - smoothed_check(r - 1e-6, h, tau)) / 2e-6;
        let RowUpdate::Fit { w: got_w, target } = majorizer_row(&rows, &[r], 0, h, tau) else {
            panic!("a least-squares row");
        };
        assert_eq!(got_w, w);
        let want = y - r + 2.0 * h * slope;
        assert!(
            (target - want).abs() <= 1e-8,
            "r {r}: {target} against {want}"
        );
    }
}

/// The stream's row: inside the band a least-squares row at `y + 2h(τ −
/// ½)`, outside it a nudge `w·2h·ψ(r)`.
#[test]
fn the_streams_row_is_a_band_row_or_a_nudge() {
    let (h, tau, y, w) = (3.0, 0.75, 10.0, 2.5);
    let (buf, ages) = one_row([0.1, 0.2], y, w);
    let rows = Rows::new(&buf, &ages, 2);
    let RowUpdate::Fit { w: fw, target } = stream_row(&rows, &[0.3], 0, h, tau) else {
        panic!("inside the band");
    };
    assert_eq!((fw, target), (w, y + 2.0 * h * (tau - 0.5)));
    for (r, s) in [(4.0, tau), (-4.0, tau - 1.0)] {
        let RowUpdate::Nudge { nudge } = stream_row(&rows, &[r], 0, h, tau) else {
            panic!("outside the band");
        };
        assert_eq!(nudge, w * 2.0 * h * s);
    }
}

fn cfg(ridge: f64) -> RobustCfg {
    RobustCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Lam(1.0),
        loss: RobustLoss::Quantile { tau: 0.75 },
        ridge,
        standardize: false,
        min_weight: 0.0,
        solve_every: 0.0,
        max_rows_between_solves: u32::MAX,
        solve_share: None,
        quantile_eps: 0.2,
        huber_scale: HuberScale::Mad,
        huber_scale_decay: None,
        feature_sets: Vec::new(),
        ridge_scale: false,
        emit_scale: false,
        refit_rows: 0,
        mad_anchor: true,
        min_weight_holds_learning: false,
    }
}

/// `Robust::damped` on one row inside the band of `h = 1` at `τ = 0.75`,
/// from `b = 0` toward a step `δ` of the intercept alone, so the row's
/// fitted value moves by `δ` and `F(s) = ρ_h(r − s) − ρ_h(r)` is the
/// parabola `(s − s*)²/4 − s*²/4`, `s* = r + ½`.
fn damped_one_row(r: f64, delta: f64) -> Option<Vec<f64>> {
    let m = Robust::new(cfg(0.0)).unwrap();
    let (buf, ages) = one_row([0.3, -0.2], r, 1.0);
    let rows = Rows::new(&buf, &ages, 2);
    m.damped(
        0,
        &rows,
        &[1.0],
        &[r],
        &[0.0; 3],
        vec![delta, 0.0, 0.0],
        1.0,
        0.75,
    )
}

/// The full step stands where it lowers `F` or raises it by no more than
/// `F`'s rounding, `64 ε` times the size of its terms; past that it is
/// halved until `F` falls, and where a halving leaves `F` exactly where it
/// was, no step along it lowers `F` and the fit stands (`None`).
#[test]
fn the_damped_step_stands_within_fs_rounding_and_is_halved_past_it() {
    let delta = 1e-6;
    // A step that lowers F: `s*` at `δ`.
    assert_eq!(
        damped_one_row(delta - 0.5, delta),
        Some(vec![delta, 0.0, 0.0])
    );
    // `s* = δ/2 − η/2`: the full step raises F by `δ·η/4` against terms of
    // size `δ/2`. At `η = 4e-15` that is under `64 ε · δ/2`, and the step
    // stands; at `η = 4e-10` it is not, and half the step lowers F.
    let r = |eta: f64| (delta - 1.0) / 2.0 - eta / 2.0;
    assert_eq!(damped_one_row(r(4e-15), delta), Some(vec![delta, 0.0, 0.0]));
    assert_eq!(
        damped_one_row(r(4e-10), delta),
        Some(vec![delta / 2.0, 0.0, 0.0])
    );
    // `s* = δ/4` at `δ = 1/2`, in doubles exactly: the full step raises F
    // by `1/32`, half of it leaves F exactly as it was, so no halving
    // lowers it.
    assert_eq!(damped_one_row(-0.375, 0.5), None);
}

/// The penalty's change `Σ s²·(c − b)(c + b)` is in `F`, scaled by the
/// band weight times the ridge over `4h`, and so is its size: two slopes
/// moving by `2^-46` from `±1`, the row's fitted value unmoved (its
/// features 0), change the penalty by `2^-91` against terms of `2^-44`,
/// within `F`'s rounding, and the step stands; by `2^-40` they change it
/// by `2^-79` against `2^-38`, past it, every halving raises it, and the
/// fit stands.
#[test]
fn the_penalty_and_its_size_are_in_the_damped_steps_change() {
    let mut m = Robust::new(cfg(2.0)).unwrap();
    m.wj[0] = 1.0;
    let (buf, ages) = one_row([0.0, 0.0], 0.5, 1.0);
    let rows = Rows::new(&buf, &ages, 2);
    let b = [0.0, 1.0, -1.0];
    let step = |d: f64| vec![0.0, 1.0 + d, -1.0 + d];
    let within = 2f64.powi(-46);
    assert_eq!(
        m.damped(0, &rows, &[1.0], &[0.5], &b, step(within), 1.0, 0.75),
        Some(step(within))
    );
    let past = 2f64.powi(-40);
    assert_eq!(
        m.damped(0, &rows, &[1.0], &[0.5], &b, step(past), 1.0, 0.75),
        None
    );
}
