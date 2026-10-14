//! [`super::colstats`] (docs/PLAN.md task 256): every target's quadratic
//! forms the bits of the one-target sums they replace, written here as
//! `ridge_fits` wrote them before, whatever targets share the call.

use super::colstats::{lower_quad, sym_quad};
use crate::SplitMix64;
use faer::Mat;

fn normal(rng: &mut SplitMix64) -> f64 {
    let u = 1.0 - rng.uniform();
    let v = rng.uniform();
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

/// The edges of the tiles' rows (16, 8, 4 and 1) and past them.
const ORDERS: [usize; 13] = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 21, 70];
/// The edges of the tiles' columns (8, 4 and 1 on aarch64, 4 and 1
/// elsewhere) and past them.
const WIDTHS: [usize; 10] = [1, 2, 3, 4, 5, 8, 9, 12, 13, 31];

fn column(x: &[f64], t: usize, c: usize) -> Vec<f64> {
    x.chunks_exact(t).map(|row| row[c]).collect()
}

/// `b' C b` as each target summed it alone.
fn sym_alone(c: &[f64], b: &[f64]) -> f64 {
    let n = b.len();
    (0..n)
        .map(|p| b[p] * (0..n).map(|q| c[p * n + q] * b[q]).sum::<f64>())
        .sum()
}

/// `‖L⁻¹ m‖²` as each target summed it alone.
fn lower_alone(inv: &Mat<f64>, m: &[f64]) -> f64 {
    (0..m.len())
        .map(|j| (0..=j).map(|i| inv[(j, i)] * m[i]).sum::<f64>().powi(2))
        .sum()
}

#[test]
fn each_targets_forms_are_its_own_sums_to_the_bit() {
    let mut rng = SplitMix64::new(25);
    for &n in &ORDERS {
        // Entries of mixed signs and scales, and exact zeros (a dropped
        // slot's 0 slope), so a sum's order shows in its last bits.
        let c: Vec<f64> = (0..n * n)
            .map(|_| normal(&mut rng) * 10f64.powi((rng.uniform() * 6.0) as i32 - 3))
            .collect();
        let inv = Mat::from_fn(n, n, |i, j| {
            if j <= i {
                normal(&mut rng) * 10f64.powi((rng.uniform() * 4.0) as i32 - 2)
            } else {
                f64::NAN
            }
        });
        for &t in &WIDTHS {
            let b: Vec<f64> = (0..n * t)
                .map(|i| if i % 7 == 3 { 0.0 } else { normal(&mut rng) })
                .collect();
            let sym = sym_quad(&c, n, &b, t);
            let low = lower_quad(&inv, &b, t);
            for col in 0..t {
                let bc = column(&b, t, col);
                assert_eq!(
                    sym[col].to_bits(),
                    sym_alone(&c, &bc).to_bits(),
                    "n={n} t={t} c={col}"
                );
                assert_eq!(
                    low[col].to_bits(),
                    lower_alone(&inv, &bc).to_bits(),
                    "n={n} t={t} c={col}"
                );
            }
        }
    }
}

#[test]
fn a_sum_of_negative_zeros_keeps_its_sign_as_the_fold_from_minus_zero_does() {
    let c = vec![-0.0, 1.0, 1.0, -0.0];
    let b = vec![0.0, -0.0];
    let alone = sym_alone(&c, &b);
    assert_eq!(sym_quad(&c, 2, &b, 1)[0].to_bits(), alone.to_bits());
    let inv = Mat::from_fn(2, 2, |i, j| if j <= i { -0.0 } else { 0.0 });
    assert_eq!(
        lower_quad(&inv, &b, 1)[0].to_bits(),
        lower_alone(&inv, &b).to_bits()
    );
    assert_eq!(sym_quad(&[], 0, &[], 3), vec![-0.0; 3]);
}
