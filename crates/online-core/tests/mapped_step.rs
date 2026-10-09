//! The step and the projection of `sgd`'s and `pa`'s fit once it is held in
//! the caller's units, against the module docs' equations (docs/PLAN.md
//! tasks 206 and 220).
//!
//! Past the warm-up each row is standardized against the features'
//! moments with the row admitted at unit weight, `z_i = (x_i − m_i) / s_i`
//! (`x_i / s_i` through the origin, `s_i` the root of the raw second
//! moment), and the row's step `Δβ` in those coordinates is mapped into
//! the caller's units, `Δb_i = Δβ_i / s_i` and `Δb_0 = Δβ_0 − Σ m_i Δb_i`.
//! A box or a sum is then imposed by the projection in the metric of the
//! row's standardized coordinates, `min Σ s_i² (c_i − b̂_i)²` over the
//! caller's bounds, the intercept keeping its standardized value `b_0 + Σ
//! m_i b_i`. Here the moments are weighted sums over the rows kept by the
//! test, the projection's multiplier is found by bisection, and each fit
//! is held to them row by row, every target and every coefficient.
//!
//! Every stream decays by a literal factor at unit clock steps.

use online_core::*;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

const LAM: f64 = 0.95;

/// The constraints each model is held under: none, a box that binds both
/// slopes of the first target, a sum alone, and a simplex.
fn constraints() -> Vec<(&'static str, Option<Constraint>)> {
    let inf = f64::INFINITY;
    vec![
        ("none", None),
        (
            "a box",
            Some(Constraint {
                lo: vec![-inf, -1.0],
                hi: vec![1.2, inf],
                sum: None,
            }),
        ),
        (
            "a sum",
            Some(Constraint {
                lo: vec![-inf, -inf],
                hi: vec![inf, inf],
                sum: Some(0.5),
            }),
        ),
        (
            "a simplex",
            Some(Constraint {
                lo: vec![0.0, 0.0],
                hi: vec![inf, inf],
                sum: Some(1.5),
            }),
        ),
    ]
}

/// A row: two features at levels of their own, two targets (the second
/// null one row in seven), and a weight of 1, 2.5, 0.5 or 0.
type Row = ([f64; 2], [Option<f64>; 2], f64);

fn rows(n: usize) -> Vec<Row> {
    let mut s = 220u64;
    (0..n)
        .map(|i| {
            let x = [3.0 + 2.0 * lcg(&mut s), -1.0 + 0.5 * lcg(&mut s)];
            let y0 = 1.0 + 2.0 * x[0] - 1.5 * x[1] + 0.2 * lcg(&mut s);
            let y1 = -0.5 + 0.7 * x[0] + 0.1 * lcg(&mut s);
            let w = match i % 5 {
                1 => 2.5,
                2 => 0.5,
                3 if i > 0 => 0.0,
                _ => 1.0,
            };
            (x, [Some(y0), (i % 7 != 3).then_some(y1)], w)
        })
        .collect()
}

/// The row's map: each feature's mean and scale with the row admitted at
/// unit weight after the decay, over the earlier rows at their weights
/// decayed since, and `z`, the intercept's 1 first where there is one.
/// Through the origin the mean is 0 and the scale the root of the raw
/// second moment; a scale of 0 is read as 1.
type Map = (Vec<f64>, Vec<f64>, Vec<f64>);

fn map(history: &[([f64; 2], f64)], x: &[f64; 2], off: usize) -> Map {
    let n = history.len();
    let mut weights: Vec<f64> = history
        .iter()
        .enumerate()
        .map(|(r, (_, w))| w * LAM.powi((n - r) as i32))
        .collect();
    weights.push(1.0);
    let total: f64 = weights.iter().sum();
    let value = |r: usize, f: usize| if r < n { history[r].0[f] } else { x[f] };
    let (mut m, mut sc, mut z) = (vec![0.0; 2], vec![1.0; 2], vec![1.0; off]);
    for f in 0..2 {
        let mean_of = |g: &dyn Fn(f64) -> f64| -> f64 {
            let sum: f64 = weights
                .iter()
                .enumerate()
                .map(|(r, w)| w * g(value(r, f)))
                .sum();
            sum / total
        };
        let spread = if off == 1 {
            m[f] = mean_of(&|v| v);
            mean_of(&|v| (v - m[f]) * (v - m[f]))
        } else {
            mean_of(&|v| v * v)
        };
        sc[f] = if spread > 0.0 { spread.sqrt() } else { 1.0 };
        z.push((x[f] - m[f]) / sc[f]);
    }
    (m, sc, z)
}

/// `Δβ` mapped into the caller's units and added to `b`.
fn add_mapped(b: &[f64], dbeta: &[f64], m: &[f64], sc: &[f64], off: usize) -> Vec<f64> {
    let mut out = b.to_vec();
    let mut shift = 0.0;
    for f in 0..2 {
        let db = dbeta[off + f] / sc[f];
        out[off + f] += db;
        shift += m[f] * db;
    }
    if off == 1 {
        out[0] += dbeta[0] - shift;
    }
    out
}

/// The slopes of `b` projected on `c` in the metric `Σ s_i² (c_i − b_i)²`:
/// `c_i = clamp(b_i − μ / s_i², lo_i, hi_i)`, with `μ = 0` for a box and
/// otherwise the root of `Σ c_i(μ) = sum`, which falls with `μ`, found by
/// bisection; the intercept takes `Σ m_i (b_i − c_i)`.
fn projected(b: &[f64], c: &Constraint, m: &[f64], sc: &[f64], off: usize) -> Vec<f64> {
    let at = |mu: f64| -> Vec<f64> {
        (0..2)
            .map(|f| (b[off + f] - mu / (sc[f] * sc[f])).clamp(c.lo[f], c.hi[f]))
            .collect()
    };
    let slopes = match c.sum {
        None => at(0.0),
        Some(s) => {
            let g = |mu: f64| at(mu).iter().sum::<f64>() - s;
            let (mut lo, mut hi) = (-1.0, 1.0);
            while g(lo) < 0.0 {
                lo *= 2.0;
            }
            while g(hi) > 0.0 {
                hi *= 2.0;
            }
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if g(mid) > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            at(0.5 * (lo + hi))
        }
    };
    let mut out = b.to_vec();
    for f in 0..2 {
        out[off + f] = slopes[f];
        if off == 1 {
            out[0] += m[f] * (b[1 + f] - slopes[f]);
        }
    }
    out
}

fn close(got: &[f64], want: &[f64]) -> bool {
    got.iter()
        .zip(want)
        .all(|(g, w)| (g - w).abs() <= 1e-9 * w.abs().max(1.0))
}

/// `x̃ᵀb`.
fn dot(b: &[f64], x: &[f64; 2], off: usize) -> f64 {
    let slopes: f64 = (0..2).map(|f| b[off + f] * x[f]).sum();
    if off == 1 { b[0] + slopes } else { slopes }
}

/// The two models behind what the test reads.
trait Held {
    fn step(&mut self, x: &[f64], y: &[Option<f64>], d: f64, w: f64) -> Vec<f64>;
    fn coef(&self) -> Vec<Vec<f64>>;
    fn switched(&self) -> bool;
    fn spread(&self) -> Vec<f64>;
}

macro_rules! held {
    ($ty:ty) => {
        impl Held for $ty {
            fn step(&mut self, x: &[f64], y: &[Option<f64>], d: f64, w: f64) -> Vec<f64> {
                OnlineModel::step(self, x, y, d, w).pred
            }
            fn coef(&self) -> Vec<Vec<f64>> {
                self.coefficients()
            }
            fn switched(&self) -> bool {
                self.warmup().switched()
            }
            fn spread(&self) -> Vec<f64> {
                self.target_variance()
            }
        }
    };
}
held!(Pa);
held!(Sgd);

/// What one target's step after the switch reads -- the coefficients
/// before it, the row's prediction, the target, `z`, the scales, the
/// row's weight and the target's spread as the row arrives -- and the
/// step `Δβ` in the row's standardized coordinates, or `None` where the
/// row teaches it nothing.
struct Seen<'a> {
    b: &'a [f64],
    p: f64,
    y: f64,
    z: &'a [f64],
    sc: &'a [f64],
    w: f64,
    spread: f64,
}

/// A model held against the equations: before the switch it is only fed;
/// after it each row's prediction is `x̃ᵀb` and its coefficients are the
/// ones before it plus the step `step_of` gives, mapped and projected as
/// the module docs write it. Returns how many steps were taken and how
/// many of them the projection moved.
fn against_the_equations(
    what: &str,
    model: &mut dyn Held,
    constraint: Option<&Constraint>,
    off: usize,
    step_of: &dyn Fn(&Seen) -> Option<Vec<f64>>,
) -> (usize, usize) {
    let mut history: Vec<([f64; 2], f64)> = Vec::new();
    let (mut stepped, mut moved) = (0, 0);
    for (i, (x, y, w)) in rows(260).into_iter().enumerate() {
        let d = if i == 0 { 0.0 } else { 1.0 };
        if !model.switched() {
            model.step(&x, &y, d, w);
            history.push((x, w));
            continue;
        }
        let (before, spread) = (model.coef(), model.spread());
        let (m, sc, z) = map(&history, &x, off);
        let mut want = before.clone();
        for (j, b) in before.iter().enumerate() {
            let Some(yj) = y[j].filter(|_| w > 0.0) else {
                continue;
            };
            let seen = Seen {
                b,
                p: dot(b, &x, off),
                y: yj,
                z: &z,
                sc: &sc,
                w,
                spread: spread.get(j).copied().unwrap_or(f64::NAN),
            };
            let Some(dbeta) = step_of(&seen) else {
                continue;
            };
            stepped += 1;
            let stepped_b = add_mapped(b, &dbeta, &m, &sc, off);
            want[j] = match constraint {
                Some(c) => {
                    let p = projected(&stepped_b, c, &m, &sc, off);
                    moved += usize::from(!close(&p, &stepped_b));
                    p
                }
                None => stepped_b,
            };
        }
        let pred = model.step(&x, &y, d, w);
        let got = model.coef();
        for j in 0..2 {
            assert!(
                close(&got[j], &want[j]),
                "{what}: row {i}, target {j}: {:?} against {:?}",
                got[j],
                want[j]
            );
            let p = dot(&before[j], &x, off);
            assert!(
                (pred[j] - p).abs() <= 1e-12 * p.abs().max(1.0),
                "{what}: row {i}, target {j}: predicted {} against {p}",
                pred[j]
            );
        }
        history.push((x, w));
    }
    (stepped, moved)
}

/// `pa`'s step after the switch (the module doc): `p = x̃ᵀb`, the loss
/// outside the tube `eps σ_y`, `σ_y²` the target's spread as the row
/// arrives, `s = ‖z‖²`, `tau` by the mode, `Δβ = min(w, 1) tau sign(y − p)
/// z`; then mapped and projected. Under each mode, each constraint, with
/// and without an intercept.
#[test]
fn pa_past_the_switch_steps_and_projects_as_written() {
    for mode in [PaMode::Pa, PaMode::Pa1, PaMode::Pa2] {
        for fit_intercept in [true, false] {
            for (name, constraint) in constraints() {
                let (eps, c) = (0.05, 0.5);
                let mut m = Pa::new(PaCfg {
                    n_features: 2,
                    n_targets: 2,
                    fit_intercept,
                    decay: Decay::Lam(LAM),
                    mode,
                    c,
                    eps,
                    min_weight: 0.0,
                    constraint: constraint.clone(),
                    standardize: true,
                })
                .unwrap();
                let what = format!("pa {mode:?}, intercept {fit_intercept}, {name}");
                let step_of = |s: &Seen| {
                    let tube = if s.spread > 0.0 && s.spread.is_finite() {
                        eps * s.spread.sqrt()
                    } else {
                        0.0
                    };
                    let err = s.y - s.p;
                    let norm: f64 = s.z.iter().map(|v| v * v).sum();
                    let loss = (err.abs() - tube).max(0.0);
                    if loss == 0.0 || norm <= 0.0 {
                        return None;
                    }
                    let tau = s.w.min(1.0)
                        * match mode {
                            PaMode::Pa => loss / norm,
                            PaMode::Pa1 => (loss / norm).min(c),
                            PaMode::Pa2 => loss / (norm + 0.5 / c),
                        };
                    Some(s.z.iter().map(|v| tau * err.signum() * v).collect())
                };
                let off = usize::from(fit_intercept);
                let (stepped, moved) =
                    against_the_equations(&what, &mut m, constraint.as_ref(), off, &step_of);
                assert!(stepped > 150, "{what}: {stepped} steps");
                if constraint.is_some() {
                    assert!(moved > 50, "{what}: the projection moved {moved}");
                }
            }
        }
    }
}

/// `sgd`'s step after the switch under the squared loss (the module doc):
/// `d = p − y`, `g_i = d z_i w + l2 s_i b_i` and `g_0 = d w`, each clipped,
/// `Δβ = −lr g`; then mapped and projected. Each constraint, with and
/// without an intercept, the clip binding now and then.
#[test]
fn sgd_past_the_switch_steps_and_projects_as_written() {
    for fit_intercept in [true, false] {
        for (name, constraint) in constraints() {
            let (lr, l2, clip) = (0.05, 0.01, 2.0);
            let off = usize::from(fit_intercept);
            let mut m = Sgd::new(SgdCfg {
                n_features: 2,
                n_targets: 2,
                fit_intercept,
                decay: Decay::Lam(LAM),
                loss: SgdLoss::Squared,
                learning_rate: lr,
                schedule: LearningRate::Constant,
                l2,
                min_weight: 0.0,
                clip_gradient: clip,
                constraint: constraint.clone(),
                standardize: true,
                strict_binary: false,
            })
            .unwrap();
            let what = format!("sgd, intercept {fit_intercept}, {name}");
            let clipped = std::cell::Cell::new(0usize);
            let step_of = |s: &Seen| {
                let d = s.p - s.y;
                let dbeta = (0..s.z.len())
                    .map(|i| {
                        let g = if i < off {
                            d * s.w
                        } else {
                            d * s.z[i] * s.w + l2 * (s.sc[i - off] * s.b[i])
                        };
                        clipped.set(clipped.get() + usize::from(g.abs() > clip));
                        -lr * g.clamp(-clip, clip)
                    })
                    .collect();
                Some(dbeta)
            };
            let (stepped, moved) =
                against_the_equations(&what, &mut m, constraint.as_ref(), off, &step_of);
            assert!(stepped > 150, "{what}: {stepped} steps");
            assert!(
                clipped.get() > 10,
                "{what}: the clip bound {} times",
                clipped.get()
            );
            if constraint.is_some() {
                assert!(moved > 50, "{what}: the projection moved {moved}");
            }
        }
    }
}
