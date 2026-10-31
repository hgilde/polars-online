//! The projected weight a readiness notice names ([`super::projected_figure`])
//! and the ceiling it is read from ([`super::settled_weight`]); moved out of
//! `stream.rs` under the 250 KB cap a source file is held to.

use super::{projected_figure, settled_frac, settled_weight};
use online_core::Decay;

/// A notice gives a projected weight only when it is a positive, finite
/// number, in four decimals in the everyday range and four significant
/// figures outside it (task 208: a sparse target read "-0.0000", a heavy
/// row a ninety-digit negative).
#[test]
fn a_projection_is_printed_only_when_it_is_a_positive_weight() {
    for bad in [
        0.0,
        -0.0,
        -1e-9,
        -2.4e88,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        assert_eq!(projected_figure(bad), None, "{bad}");
    }
    assert_eq!(projected_figure(7.725).as_deref(), Some("7.7250"));
    assert_eq!(projected_figure(1e-3).as_deref(), Some("0.0010"));
    assert_eq!(projected_figure(3e-7).as_deref(), Some("3.0000e-7"));
    assert_eq!(projected_figure(1e90).as_deref(), Some("1.0000e90"));
    assert_eq!(projected_figure(999_999.0).as_deref(), Some("999999.0000"));
    assert_eq!(projected_figure(1e6).as_deref(), Some("1.0000e6"));
}

/// `x` moved by `k` ulp: a libm that rounds the other way, or a step
/// further off.
fn nudged(x: f64, k: i64) -> f64 {
    f64::from_bits((x.to_bits() as i64 + k) as u64)
}

/// Task 273: a target present on the first row alone, at half-life 5,
/// read 28 rows on. Its weight is that row's, decayed: `W = λ²⁸` by 28
/// multiplies, and `1 − s` is the same `2^(−28/5)` from one `exp2`, so
/// `W − w₁(1 − s)` is 0 and the computed one is their roundings. With
/// the wheel's `exp2` 1 ulp high it came out +5.7e-17, and the notice
/// advised lowering `min_weight` below "5.6858e-17"; on macOS's libm it
/// came out at or below 0, and the notice gave no figure. Within its
/// rounding the difference is 0 on every libm, here moved by up to 2
/// ulp either way.
#[test]
fn a_difference_within_its_rounding_projects_no_ceiling() {
    for k in [-2, -1, 0, 1, 2] {
        let lam = nudged((-1.0f64 / 5.0).exp2(), k);
        let mut weight = 1.0;
        for _ in 0..28 {
            weight *= lam;
        }
        let settled = 1.0 - nudged((-28.0f64 / 5.0).exp2(), k);
        let ceiling = settled_weight(weight, 1.0, settled, 29.0);
        assert_eq!(
            ceiling,
            0.0,
            "k = {k}: {weight:e} against {:e}",
            1.0 - settled
        );
        assert_eq!(projected_figure(ceiling), None, "k = {k}");
    }
    // The same by literals, no libm: `W` one ulp above `1 − s`.
    let settled = 0.98;
    let weight = nudged(1.0 - settled, 1);
    assert_eq!(settled_weight(weight, 1.0, settled, 1.0), 0.0);
}

/// The rounding band does not swallow a real ceiling, however small: a
/// regular stream of rows weighing 1e-9, one clock unit apart at
/// half-life 5, tops out at `w/(1 − 2^(−1/5)) = 7.725e-9`, which the
/// projection reads from the second row on (the doc above).
#[test]
fn a_small_real_ceiling_keeps_its_figure() {
    let (w1, h) = (1e-9, 5.0);
    let lam = (-1.0f64 / h).exp2();
    let want = w1 / (1.0 - lam);
    let mut weight = 0.0;
    for n in 1..=60 {
        weight = lam * weight + w1;
        if n < 2 {
            continue;
        }
        let settled = settled_frac(Decay::Halflife(h), (n - 1) as f64);
        let ceiling = settled_weight(weight, w1, settled, n as f64);
        assert!(
            (ceiling - want).abs() <= 1e-12 * want,
            "row {n}: {ceiling:e}"
        );
        assert_eq!(
            projected_figure(ceiling).as_deref(),
            Some("7.7250e-9"),
            "row {n}"
        );
    }
}
