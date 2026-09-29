//! The critical value of Wied & Galeano's sequential detector
//! (`corrchange(kind = "sequential")`, docs/PLAN.md task 114).
//!
//! Wied & Galeano (2013; SFB 823 Discussion Paper 12/2012, Eq. 7) show that
//! under the null their detector crosses the boundary `c·w(k/m)` with
//! limiting probability
//!
//! ```text
//! P( (T/(1+T))^{1/2−γ} · Z_γ > c ),   Z_γ = sup_{0<s≤1} |W(s)| / s^γ,
//! ```
//!
//! `W` a standard Brownian motion and `0 ≤ γ < 1/2`, so `c(α) = (T/(1+T))^
//! {1/2−γ}·q_γ(1 − α)` with `q_γ` the quantile function of `Z_γ`. This
//! module computes `q_γ`.
//!
//! # γ = 0: a series
//!
//! `Z_0` is the supremum of `|W|` on `[0, 1]`, whose law is the series
//! (Feller 1951; Borodin & Salminen, 1.1.4)
//!
//! ```text
//! P(Z_0 ≤ x) = (4/π) Σ_{k≥0} (−1)^k/(2k+1) · exp(−(2k+1)²π²/(8x²)),
//! ```
//!
//! 0.95 at `x = 2.2414`, so at `T = 1` the 5 % value is `√½·2.2414 = 1.5849`.
//!
//! # γ > 0: a diffusion with an absorbing boundary
//!
//! There is no series; W&G simulate 10,000 paths on a grid of 10,000
//! points (their Table 1). Here the law is solved instead. With `t = −ln s`,
//! `U(t) = e^{t/2}·W(e^{−t})` is a stationary Ornstein–Uhlenbeck process,
//! `dU = −U/2 dt + dB` with `U(0) = W(1) ~ N(0, 1)` (its covariance is
//! `e^{−|t−t'|/2}`), and `|W(s)|/s^γ = e^{−βt}|U(t)|` with `β = 1/2 − γ`. So
//! `Z_γ ≤ c` exactly when `|U(t)| ≤ c·e^{βt}` for every `t ≥ 0`: an OU path
//! inside a widening corridor. In `y = U/(c·e^{βt})` the corridor is `[−1, 1]`
//! and the density `q(t, y)` of the paths still inside it solves
//!
//! ```text
//! q_t = ∂_y[ D(t)·q_y + κ·y·q ],   D(t) = e^{−2βt}/(2c²),   κ = 1 − γ,
//! q(t, ±1) = 0,                    q(0, y) = c·φ(c·y),
//! ```
//!
//! and `P(Z_γ ≤ c)` is the mass left as `t → ∞`. `D` shrinks and the drift
//! gathers the mass towards 0, so it settles: the solve stops once the
//! spread the two balance at, `√(D/κ)`, is under 1/8 of the corridor, and
//! four relaxation times `1/κ` after that. The absorption is continuous,
//! where a simulation on a grid misses the crossings between its points and
//! so reads `Z_γ` low. That is why these values sit above W&G's Table 1 in
//! 11 of its 12 cells, and within 0.03 of it in all; the twelfth is below
//! by 0.015, in a row of theirs that is off the exact `T`-scaling by 0.027,
//! the scatter of their 10,000 paths.
//!
//! The space grid has 400 cells, with Scharfetter–Gummel fluxes, which stay
//! stable and positive whatever the ratio of drift to diffusion; time runs
//! by Crank–Nicolson in steps of 0.002 after four implicit Euler quarter
//! steps, which damp the jump between the initial density and the boundary.
//! At `γ = 0` the solve matches the series to 3e-6 at every `x` the tests
//! read, and halving both steps moves it by a quarter of that.
//!
//! The quantile is found by regula falsi (Illinois) on the solve, bracketed
//! from `Z_γ ≥ Z_0`, and kept per `(α, γ)` for the life of the process, so a
//! bank of many groups computes it once. The arithmetic calls `exp`, whose
//! last bits differ between platforms' libms, so the value may differ in its
//! last bits too; it is recomputed when a state loads, never stored (docs
//! "No libm in the state").

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// `P(sup_{0≤s≤1} |W(s)| ≤ x)`, the series above.
pub fn sup_abs_bm_cdf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    let a = std::f64::consts::PI * std::f64::consts::PI / (8.0 * x * x);
    let mut sum = 0.0;
    for k in 0..400u32 {
        let n = f64::from(2 * k + 1);
        let term = (-(n * n) * a).exp() / n;
        sum += if k % 2 == 0 { term } else { -term };
        if term < 1e-18 {
            break;
        }
    }
    (4.0 / std::f64::consts::PI * sum).clamp(0.0, 1.0)
}

/// The `p` quantile of `sup_{0≤s≤1} |W(s)|`, by bisection on the series.
pub fn sup_abs_bm_quantile(p: f64) -> f64 {
    if !(p > 0.0 && p < 1.0) {
        return f64::NAN;
    }
    let (mut lo, mut hi) = (0.0f64, 20.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if sup_abs_bm_cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// `B(z) = z/(e^z − 1)`, the Bernoulli function of the Scharfetter–Gummel
/// flux, with its limit 1 at 0.
fn bernoulli(z: f64) -> f64 {
    if z.abs() < 1e-8 {
        1.0 - 0.5 * z
    } else {
        z / z.exp_m1()
    }
}

/// Cells of the space grid and the Crank–Nicolson step (see the module docs).
const CELLS: usize = 400;
const DT: f64 = 2e-3;

/// `P(Z_γ ≤ c)`, `Z_γ = sup_{0<s≤1} |W(s)|/s^γ`, by the absorbed diffusion
/// of the module docs. `0 ≤ γ < 1/2`.
pub fn boundary_cdf(c: f64, gamma: f64) -> f64 {
    if c.is_nan() || !(0.0..0.5).contains(&gamma) {
        return f64::NAN;
    }
    if c <= 0.0 {
        return 0.0;
    }
    let (beta, kappa) = (0.5 - gamma, 1.0 - gamma);
    let m = CELLS;
    let h = 2.0 / m as f64;
    // Interior nodes y_1 .. y_{m-1}; the ends are the absorbing boundary.
    let n = m - 1;
    let mut q: Vec<f64> = (1..m)
        .map(|i| {
            let y = -1.0 + h * i as f64;
            c * (-0.5 * (c * y) * (c * y)).exp() / (2.0 * std::f64::consts::PI).sqrt()
        })
        .collect();
    // Face j sits between nodes j and j+1, at y = −1 + h(j + 1/2).
    let faces: Vec<f64> = (0..m).map(|j| -1.0 + h * (j as f64 + 0.5)).collect();
    let (mut a, mut b) = (vec![0.0; m], vec![0.0; m]);
    let (mut lo, mut di, mut up) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut rhs = vec![0.0; n];
    let mut cp = vec![0.0; n];
    let spread = (1.0 / (2.0 * c * c * kappa)).sqrt();
    let t_end = (8.0 * spread).ln().max(0.0) / beta + 4.0 / kappa;

    let mut step = |q: &mut Vec<f64>, t: f64, dt: f64, theta: f64| {
        // The operator at the stage time t + θ·dt.
        let d = (-2.0 * beta * (t + theta * dt)).exp() / (2.0 * c * c);
        for j in 0..m {
            let p = -kappa * faces[j] * h / d;
            a[j] = d / h * bernoulli(-p);
            b[j] = d / h * bernoulli(p);
        }
        // Row i (node i + 1): q_{i-1} at a_{i}/h, q_i at −(a_{i+1} + b_{i})/h,
        // q_{i+1} at b_{i+1}/h, faces indexed as above.
        for i in 0..n {
            lo[i] = a[i] / h;
            di[i] = -(a[i + 1] + b[i]) / h;
            up[i] = b[i + 1] / h;
        }
        for i in 0..n {
            let mut aq = di[i] * q[i];
            if i > 0 {
                aq += lo[i] * q[i - 1];
            }
            if i + 1 < n {
                aq += up[i] * q[i + 1];
            }
            rhs[i] = q[i] + (1.0 - theta) * dt * aq;
        }
        // (I − θ·dt·A) q' = rhs, by the Thomas algorithm.
        let (sub, main, sup) = (
            |i: usize| -theta * dt * lo[i],
            |i: usize| 1.0 - theta * dt * di[i],
            |i: usize| -theta * dt * up[i],
        );
        let mut denom = main(0);
        cp[0] = sup(0) / denom;
        q[0] = rhs[0] / denom;
        for i in 1..n {
            denom = main(i) - sub(i) * cp[i - 1];
            cp[i] = sup(i) / denom;
            q[i] = (rhs[i] - sub(i) * q[i - 1]) / denom;
        }
        for i in (0..n - 1).rev() {
            q[i] -= cp[i] * q[i + 1];
        }
    };

    let mut t = 0.0;
    for _ in 0..4 {
        step(&mut q, t, 0.25 * DT, 1.0);
        t += 0.25 * DT;
    }
    while t < t_end {
        step(&mut q, t, DT, 0.5);
        t += DT;
    }
    (h * q.iter().sum::<f64>()).clamp(0.0, 1.0)
}

/// The `p` quantile of `Z_γ`: the series at `γ = 0`, the solve otherwise,
/// by Illinois regula falsi bracketed from `Z_γ ≥ Z_0`. `NaN` for `p`
/// outside `(0, 1)` or `γ` outside `[0, 1/2)`.
pub fn boundary_quantile(p: f64, gamma: f64) -> f64 {
    if !(p > 0.0 && p < 1.0) || !(0.0..0.5).contains(&gamma) {
        return f64::NAN;
    }
    let q0 = sup_abs_bm_quantile(p);
    if gamma == 0.0 {
        return q0;
    }
    static CACHE: OnceLock<Mutex<HashMap<(u64, u64), f64>>> = OnceLock::new();
    let key = (p.to_bits(), gamma.to_bits());
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&v) = cache.lock().expect("not poisoned").get(&key) {
        return v;
    }
    let f = |c: f64| boundary_cdf(c, gamma) - p;
    let (mut lo, mut flo) = (q0, f(q0));
    let (mut hi, mut fhi) = (q0 + 1.0, f(q0 + 1.0));
    while fhi < 0.0 {
        (lo, flo) = (hi, fhi);
        hi += 2.0;
        fhi = f(hi);
    }
    let mut side = 0i8;
    let mut mid = hi;
    for _ in 0..100 {
        mid = (lo * fhi - hi * flo) / (fhi - flo);
        let fm = f(mid);
        if fm.abs() < 1e-11 || (hi - lo).abs() < 1e-9 {
            break;
        }
        if fm < 0.0 {
            (lo, flo) = (mid, fm);
            if side == -1 {
                fhi *= 0.5;
            }
            side = -1;
        } else {
            (hi, fhi) = (mid, fm);
            if side == 1 {
                flo *= 0.5;
            }
            side = 1;
        }
    }
    cache.lock().expect("not poisoned").insert(key, mid);
    mid
}

/// Wied & Galeano's critical value at level `alpha` for the boundary
/// exponent `gamma` and a monitoring period `ratio` times the history
/// (their `T`): `(T/(1+T))^{1/2−γ}·q_γ(1 − α)`, their Eq. 7.
pub fn sequential_crit(alpha: f64, gamma: f64, ratio: f64) -> f64 {
    if !(ratio > 0.0 && ratio.is_finite()) {
        return f64::NAN;
    }
    (ratio / (1.0 + ratio)).powf(0.5 - gamma) * boundary_quantile(1.0 - alpha, gamma)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The series is a distribution function, and its 95 % point is the
    /// tabulated 2.2414.
    #[test]
    fn the_series_is_the_law_of_the_supremum() {
        assert_eq!(sup_abs_bm_cdf(0.0), 0.0);
        assert_eq!(sup_abs_bm_cdf(12.0), 1.0);
        for x in [0.5, 1.0, 1.5, 2.0, 3.0] {
            assert!(sup_abs_bm_cdf(x) < sup_abs_bm_cdf(x + 0.05), "{x}");
        }
        assert!((sup_abs_bm_quantile(0.95) - 2.2414).abs() < 1e-4);
        assert!(sup_abs_bm_quantile(0.0).is_nan() && sup_abs_bm_quantile(1.0).is_nan());
    }

    /// At `γ = 0` the absorbed diffusion is the supremum of `|W|`, whose law
    /// the series gives exactly: the solve reproduces it to 3e-6.
    #[test]
    fn the_solve_at_gamma_zero_is_the_series() {
        for x in [0.8, 1.5, 2.0, 2.2414, 3.0, 4.0] {
            let (got, want) = (boundary_cdf(x, 0.0), sup_abs_bm_cdf(x));
            assert!((got - want).abs() < 3e-6, "x = {x}: {got} against {want}");
        }
    }

    /// `Z_γ ≥ Z_0`, and the more so the larger `γ`: at every `c` the law is
    /// ordered.
    #[test]
    fn a_larger_gamma_is_a_larger_supremum() {
        for c in [1.5, 2.5, 3.5] {
            let (z0, z25, z45) = (
                boundary_cdf(c, 0.0),
                boundary_cdf(c, 0.25),
                boundary_cdf(c, 0.45),
            );
            assert!(z0 > z25 && z25 > z45, "c = {c}: {z0} {z25} {z45}");
        }
        assert!(boundary_cdf(2.0, 0.5).is_nan() && boundary_cdf(2.0, -0.1).is_nan());
    }

    /// Wied & Galeano's Table 1, the 5 % critical values for `γ = 0, 0.25,
    /// 0.45` and `T = 0.5, 1, 2, 4`, simulated on a grid of 10,000 points
    /// from 10,000 paths. Theirs sit below these in 11 cells of 12, by up to
    /// 0.03: a grid reads the supremum low. The twelfth (`γ = 0.25`, `T =
    /// 2`) is above by 0.015, in a row of theirs off the exact `T`-scaling by
    /// 0.027, the spread 10,000 paths leave.
    #[test]
    fn the_critical_values_are_wied_and_galeanos_table_1() {
        let table = [
            (0.0, [1.2870, 1.5578, 1.8158, 1.9980]),
            (0.25, [1.8001, 1.9924, 2.1684, 2.2467]),
            (0.45, [2.6282, 2.6844, 2.7215, 2.7660]),
        ];
        for (gamma, row) in table {
            for (ratio, theirs) in [0.5, 1.0, 2.0, 4.0].into_iter().zip(row) {
                let ours = sequential_crit(0.05, gamma, ratio);
                assert!(
                    (ours - theirs).abs() < 0.035,
                    "γ = {gamma}, T = {ratio}: {ours} against {theirs}"
                );
            }
        }
        // The γ = 0 row in closed form: √(T/(1+T))·2.2414.
        assert!((sequential_crit(0.05, 0.0, 1.0) - 0.5f64.sqrt() * 2.24140).abs() < 1e-4);
    }

    /// The quantile is the solve's inverse, and a second call is the cached
    /// value, bit for bit.
    #[test]
    fn the_quantile_inverts_the_solve() {
        let q = boundary_quantile(0.9, 0.3);
        assert!((boundary_cdf(q, 0.3) - 0.9).abs() < 1e-8, "{q}");
        assert_eq!(boundary_quantile(0.9, 0.3).to_bits(), q.to_bits());
        assert!(boundary_quantile(0.9, 0.5).is_nan() && boundary_quantile(1.0, 0.2).is_nan());
        assert!(sequential_crit(0.05, 0.2, 0.0).is_nan());
    }
}
