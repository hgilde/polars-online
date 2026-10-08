//! Every readout of the filter against its recursion, and the state and
//! re-map checks beside them (docs/PLAN.md task 218). A file of its own
//! because `kalman.rs` stands near the source-size cap
//! (`tests/test_repo_hygiene.py`).

use super::*;

/// One feature's moments over the rows so far, as `z` is formed from
/// them: the mean (0 through the origin), the scale, whether the scale
/// is usable, and `E[z²]`, what the summary's mean field weighs each
/// slot's variance by (the raw second moment unstandardized).
#[derive(Clone, Copy)]
struct Moment {
    mean: f64,
    scale: f64,
    usable: bool,
    ez2: f64,
}

/// The mean of the positive numbers in `v`, 0 where there is none:
/// `share_p`'s noise over the targets that have one.
fn mean_positive(v: &[f64]) -> f64 {
    let have: Vec<f64> = v.iter().copied().filter(|s2| *s2 > 0.0).collect();
    if have.is_empty() {
        0.0
    } else {
        have.iter().sum::<f64>() / have.len() as f64
    }
}

/// `zᵀ P z` for a row-major `P`.
fn quad_of(p: &[f64], z: &[f64]) -> f64 {
    let k = z.len();
    (0..k)
        .map(|a| z[a] * (0..k).map(|c| p[a * k + c] * z[c]).sum::<f64>())
        .sum()
}

/// The filter as the module doc writes it, held in the current
/// coordinates -- `the_standardized_filter_is_its_recursion`'s replica,
/// and `the_filter_is_its_recursion`'s without `standardize` -- with
/// every readout written from its definition beside it. The moments are
/// weighted sums over the rows kept here, the mean taken from the first
/// row's value so that a column that has not moved has a variance of
/// exactly 0, as the recursion's has; after every row that moves them,
/// `b` and `P` are carried to the new moments by `A`, `b' = A b` and `P'
/// = A P Aᵀ` (task 206's re-map on every row, which the anchored filter
/// is to rounding). Written for `obs_var` and `q` unset and `min_weight`
/// 0.
struct Twin {
    cfg: KalmanCfg,
    p: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    sig2: Vec<f64>,
    wsig: Vec<f64>,
    wj: Vec<f64>,
    elapsed: Vec<f64>,
    basis: Vec<Vec<(f64, f64)>>,
    rows: Vec<(Vec<f64>, f64)>,
}

impl Twin {
    fn new(cfg: KalmanCfg) -> Self {
        assert!(cfg.obs_var.is_none() && cfg.q.is_none() && cfg.min_weight == 0.0);
        let (k, m) = (cfg.k_total(), cfg.n_targets);
        let n_p = if cfg.share_p { 1 } else { m };
        Self {
            p: vec![vec![0.0; k * k]; n_p],
            b: vec![vec![0.0; k]; m],
            sig2: vec![0.0; m],
            wsig: vec![0.0; m],
            wj: vec![0.0; m],
            elapsed: vec![0.0; n_p],
            basis: vec![Vec::new(); n_p],
            rows: Vec::new(),
            cfg,
        }
    }

    fn k(&self) -> usize {
        self.cfg.k_total()
    }

    fn off(&self) -> usize {
        usize::from(self.cfg.fit_intercept)
    }

    /// The covariance target `j` reads.
    fn pi(&self, j: usize) -> usize {
        if self.cfg.share_p { 0 } else { j }
    }

    fn moments(&self) -> Vec<Moment> {
        let total: f64 = self.rows.iter().map(|(_, w)| w).sum();
        (0..self.cfg.n_features)
            .map(|f| {
                if total <= 0.0 {
                    return Moment {
                        mean: 0.0,
                        scale: 1.0,
                        usable: !self.cfg.standardize,
                        ez2: 0.0,
                    };
                }
                let x0 = self.rows[0].0[f];
                let sum = |g: &dyn Fn(f64) -> f64| -> f64 {
                    self.rows.iter().map(|(x, w)| w * g(x[f])).sum::<f64>() / total
                };
                let mu = x0 + sum(&|x| x - x0);
                let var = sum(&|x| (x - mu).powi(2));
                let raw = sum(&|x| x * x);
                if !self.cfg.standardize {
                    Moment {
                        mean: 0.0,
                        scale: 1.0,
                        usable: true,
                        ez2: raw,
                    }
                } else {
                    let (mean, spread) = if self.cfg.fit_intercept {
                        (mu, var)
                    } else {
                        (0.0, raw)
                    };
                    let usable = spread > 0.0;
                    let scale = if usable { spread.sqrt() } else { 1.0 };
                    Moment {
                        mean,
                        scale,
                        usable,
                        ez2: spread / (scale * scale),
                    }
                }
            })
            .collect()
    }

    /// `[1, x]` in the current coordinates.
    fn z(&self, mo: &[Moment], x: &[f64]) -> Vec<f64> {
        let off = self.off();
        let mut z = vec![1.0; self.k()];
        for (f, xf) in x.iter().enumerate() {
            z[off + f] = if self.cfg.standardize {
                (xf - mo[f].mean) / mo[f].scale
            } else {
                *xf
            };
        }
        z
    }

    fn phi(&self, a: usize, d: f64) -> f64 {
        let r = &self.cfg.revert_half_life;
        Decay::Halflife(if r.len() == 1 { r[0] } else { r[a] }).factor(d)
    }

    /// What `q_a` is multiplied by for a gap of `d`: `d²`, or on a
    /// reverting slot `((1 − 2^(−d/r)) / θ)²`, `θ = ln 2 / r`.
    fn gap2(&self, a: usize, d: f64) -> f64 {
        let r = &self.cfg.revert_half_life;
        let r = if r.len() == 1 { r[0] } else { r[a] };
        if r.is_infinite() {
            d * d
        } else {
            ((1.0 - self.phi(a, d)) / (std::f64::consts::LN_2 / r)).powi(2)
        }
    }

    /// `q_a = σ² (ln 2 / h_a)²`, 0 at `h_a = inf`.
    fn q(&self, s2: f64, a: usize) -> f64 {
        let h = &self.cfg.half_life;
        let h = if h.len() == 1 { h[0] } else { h[a] };
        if h.is_infinite() {
            0.0
        } else {
            s2 * (std::f64::consts::LN_2 / h).powi(2)
        }
    }

    fn is_unsized(&self, pi: usize) -> bool {
        let k = self.k();
        (0..k).all(|a| self.p[pi][a * k + a] == 0.0)
    }

    /// The noise a readout reads: the residual variance, under
    /// `share_p` the mean over the targets that have one; NaN for none.
    fn noise_of(&self, j: usize) -> f64 {
        let s2 = if self.cfg.share_p {
            mean_positive(&self.sig2)
        } else {
            self.sig2[j]
        };
        if s2 > 0.0 && s2.is_finite() {
            s2
        } else {
            f64::NAN
        }
    }

    fn step(&mut self, x: &[f64], ys: &[Option<f64>], d: f64, w: f64) -> Vec<f64> {
        let (k, off, m) = (self.k(), self.off(), self.cfg.n_targets);
        let share = self.cfg.share_p;
        let lam = self.cfg.decay.factor(d);
        for e in &mut self.elapsed {
            *e += d;
        }
        // The transition, diagonal in the current coordinates.
        let phi: Vec<f64> = (0..k).map(|a| self.phi(a, d)).collect();
        for bj in &mut self.b {
            for a in 0..k {
                bj[a] *= phi[a];
            }
        }
        for pp in &mut self.p {
            for a in 0..k {
                for c in 0..k {
                    pp[a * k + c] *= phi[a] * phi[c];
                }
            }
        }
        let mo = self.moments();
        let z = self.z(&mo, x);
        let dot = |v: &[f64]| -> f64 { (0..k).map(|a| z[a] * v[a]).sum() };
        let want: Vec<f64> = (0..m)
            .map(|j| {
                if self.wj[j] > 0.0 {
                    dot(&self.b[j])
                } else {
                    f64::NAN
                }
            })
            .collect();
        let obs: Vec<Option<f64>> = ys.iter().map(|y| y.filter(|_| w > 0.0)).collect();
        let e2: Vec<Option<f64>> = (0..m)
            .map(|j| obs[j].map(|y| (y - dot(&self.b[j])).powi(2)))
            .collect();
        let seen: Vec<f64> = e2.iter().flatten().copied().collect();
        let shared_first = if seen.is_empty() {
            0.0
        } else {
            seen.iter().sum::<f64>() / seen.len() as f64
        };
        let shared = mean_positive(&self.sig2);
        let noise: Vec<f64> = (0..m)
            .map(|j| {
                let s2 = if share { shared } else { self.sig2[j] };
                if s2 > 0.0 {
                    s2
                } else if share {
                    shared_first
                } else {
                    e2[j].unwrap_or(0.0)
                }
            })
            .collect();
        // A row counts once in a basis, at its mean square over the
        // targets it observes that give a scale.
        if self.cfg.standardize {
            for (pi, held) in self.basis.iter_mut().enumerate() {
                let mine: Vec<f64> = (0..m)
                    .filter(|&j| share || j == pi)
                    .filter_map(|j| e2[j].filter(|v| *v > 0.0))
                    .collect();
                if !mine.is_empty() && held.len() < 3 {
                    held.push((mine.iter().sum::<f64>() / mine.len() as f64, w));
                }
            }
        }
        for pi in 0..self.p.len() {
            let informs = if share {
                obs.iter().any(Option::is_some)
            } else {
                obs[pi].is_some()
            };
            if !informs {
                continue;
            }
            let gap = self.elapsed[pi];
            self.elapsed[pi] = 0.0;
            let s2 = noise[if share { 0 } else { pi }];
            if !self.cfg.standardize && self.is_unsized(pi) {
                for a in 0..k {
                    self.p[pi][a * k + a] = self.cfg.p0 * s2;
                }
                continue;
            }
            for a in 0..k {
                if self.p[pi][a * k + a] != 0.0 {
                    self.p[pi][a * k + a] += self.q(s2, a) * self.gap2(a, gap);
                }
            }
            if self.cfg.standardize && self.basis[pi].len() == 3 {
                // The weighted median: the value with less than half the
                // weight below it and at least half at or below it.
                let held = &self.basis[pi];
                let total: f64 = held.iter().map(|(_, w)| w).sum();
                let weight_of = |keep: &dyn Fn(f64) -> bool| -> f64 {
                    held.iter().filter(|(u, _)| keep(*u)).map(|(_, w)| w).sum()
                };
                let median = held
                    .iter()
                    .map(|(v, _)| *v)
                    .filter(|&v| {
                        weight_of(&|u| u < v) < 0.5 * total && weight_of(&|u| u <= v) >= 0.5 * total
                    })
                    .fold(f64::INFINITY, f64::min);
                let r = median / 0.4549364231195728;
                for a in 0..k {
                    let ok = a < off || mo[a - off].usable;
                    if ok && self.p[pi][a * k + a] == 0.0 {
                        self.p[pi][a * k + a] = self.cfg.p0 * r;
                    }
                }
            }
        }
        let mut shared_row: Option<(Vec<f64>, f64)> = None;
        for j in 0..m {
            let pi = self.pi(j);
            let Some(y) = obs[j] else {
                self.wj[j] *= lam;
                self.wsig[j] *= lam;
                continue;
            };
            let pz: Vec<f64> = (0..k)
                .map(|a| dot(&self.p[pi][a * k..(a + 1) * k]))
                .collect();
            let s_inn = dot(&pz) + noise[j] / w;
            let err = y - dot(&self.b[j]);
            if noise[j] > 0.0 {
                for (ba, pa) in self.b[j].iter_mut().zip(&pz) {
                    *ba += pa / s_inn * err;
                }
                if share {
                    shared_row = Some((pz, s_inn));
                } else {
                    for a in 0..k {
                        for c in 0..k {
                            self.p[pi][a * k + c] -= pz[a] * pz[c] / s_inn;
                        }
                    }
                }
            }
            let aged = lam * self.wsig[j];
            self.wsig[j] = aged;
            if want[j].is_finite() {
                let r = y - want[j];
                self.sig2[j] = (aged * self.sig2[j] + w * r * r) / (aged + w);
                self.wsig[j] = aged + w;
            }
            self.wj[j] = lam * self.wj[j] + w;
        }
        if let Some((pz, s_inn)) = shared_row {
            for a in 0..k {
                for c in 0..k {
                    self.p[0][a * k + c] -= pz[a] * pz[c] / s_inn;
                }
            }
        }
        // The moments learn the row, and `b` and `P` follow them.
        for (_, wh) in &mut self.rows {
            *wh *= lam;
        }
        self.rows.push((x.to_vec(), w));
        if self.cfg.standardize && w > 0.0 {
            let after = self.moments();
            let mut am = vec![0.0f64; k * k];
            am[0] = 1.0;
            for f in 0..self.cfg.n_features {
                let a = off + f;
                am[a * k + a] = after[f].scale / mo[f].scale;
                if off == 1 {
                    am[a] = (after[f].mean - mo[f].mean) / mo[f].scale;
                }
            }
            for bj in &mut self.b {
                let old = bj.clone();
                for r in 0..k {
                    bj[r] = (0..k).map(|c| am[r * k + c] * old[c]).sum();
                }
            }
            for pp in &mut self.p {
                let ap: Vec<f64> = (0..k * k)
                    .map(|rc| {
                        (0..k)
                            .map(|t| am[rc / k * k + t] * pp[t * k + rc % k])
                            .sum()
                    })
                    .collect();
                for rc in 0..k * k {
                    pp[rc] = (0..k)
                        .map(|t| ap[rc / k * k + t] * am[rc % k * k + t])
                        .sum();
                }
            }
        }
        want
    }

    /// `P` as it stands after the last row: with the noise of the clock
    /// since the last observation, `Q D²` on the sized slots, at the
    /// noise the state holds -- diagonal in these coordinates.
    fn p_now(&self, pi: usize) -> Vec<f64> {
        let (k, d) = (self.k(), self.elapsed[pi]);
        let noise = self.noise_of(pi);
        let mut p = self.p[pi].clone();
        if d == 0.0 || noise.is_nan() || self.is_unsized(pi) {
            return p;
        }
        for a in 0..k {
            if p[a * k + a] != 0.0 {
                p[a * k + a] += self.q(noise, a) * self.gap2(a, d);
            }
        }
        p
    }

    /// `zᵀ P z + σ²_j`, or `σ²_j (zᵀ P z / σ̄² + 1)` under `share_p`;
    /// NaN until the target has a residual variance.
    fn pred_var(&self, x: &[f64]) -> Vec<f64> {
        let z = self.z(&self.moments(), x);
        (0..self.cfg.n_targets)
            .map(|j| {
                let q = quad_of(&self.p_now(self.pi(j)), &z);
                let own = self.sig2[j];
                if self.cfg.share_p {
                    let mean = mean_positive(&self.sig2);
                    if own > 0.0 && mean > 0.0 {
                        own * (q / mean + 1.0)
                    } else {
                        f64::NAN
                    }
                } else if own > 0.0 {
                    q + own
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    /// The coefficients in the caller's units, `c_i = b_i / s_i` and
    /// `c_0 = b_0 − Σ c_i m_i`.
    fn coefficients(&self) -> Vec<Vec<f64>> {
        if !self.cfg.standardize {
            return self.b.clone();
        }
        let (k, off, mo) = (self.k(), self.off(), self.moments());
        self.b
            .iter()
            .map(|b| {
                let mut c = b.clone();
                for a in off..k {
                    c[a] = b[a] / mo[a - off].scale;
                }
                if off == 1 {
                    c[0] = b[0] - (1..k).map(|a| c[a] * mo[a - 1].mean).sum::<f64>();
                }
                c
            })
            .collect()
    }

    /// `T P Tᵀ`'s diagonal, `T` the read-out above, times `σ²_j / σ̄²`
    /// under `share_p` (1 for a target with no residual variance); NaN
    /// while `P` is unsized.
    fn coef_variance(&self) -> Vec<Vec<f64>> {
        let (k, off, mo) = (self.k(), self.off(), self.moments());
        (0..self.cfg.n_targets)
            .map(|j| {
                let pi = self.pi(j);
                if self.is_unsized(pi) {
                    return vec![f64::NAN; k];
                }
                let p = self.p_now(pi);
                let mean = mean_positive(&self.sig2);
                let own = self.sig2[j];
                let ratio = if self.cfg.share_p && own > 0.0 && mean > 0.0 {
                    own / mean
                } else {
                    1.0
                };
                let mut v: Vec<f64> = (0..k)
                    .map(|a| {
                        if a < off || !self.cfg.standardize {
                            p[a * k + a]
                        } else {
                            p[a * k + a] / mo[a - off].scale.powi(2)
                        }
                    })
                    .collect();
                if self.cfg.standardize && off == 1 {
                    let t: Vec<f64> = (0..k)
                        .map(|a| {
                            if a == 0 {
                                1.0
                            } else {
                                -mo[a - 1].mean / mo[a - 1].scale
                            }
                        })
                        .collect();
                    v[0] = quad_of(&p, &t).max(0.0);
                }
                v.iter().map(|x| x * ratio).collect()
            })
            .collect()
    }

    /// The summary's `sqrt(1 + Σ_a P_aa E[z_a²] / R)`, infinite while
    /// `P` is unsized or there is no noise.
    fn summary(&self) -> Vec<f64> {
        let (k, off, mo) = (self.k(), self.off(), self.moments());
        let ez2: Vec<f64> = (0..k)
            .map(|a| if a < off { 1.0 } else { mo[a - off].ez2 })
            .collect();
        (0..self.cfg.n_targets)
            .map(|j| {
                let r = self.noise_of(j);
                if self.is_unsized(self.pi(j)) || r.is_nan() {
                    return f64::INFINITY;
                }
                let p = self.p_now(self.pi(j));
                let trace: f64 = (0..k).map(|a| p[a * k + a] * ez2[a]).sum();
                (1.0 + trace / r).sqrt()
            })
            .collect()
    }

    /// The row's `sqrt(1 + zᵀ (Φ P Φ + Q (D + d)²) z / R)`, `z` the row
    /// against the moments before it, the noise on the sized slots;
    /// infinite while `P` is unsized or there is no noise.
    fn row_stat(&self, x: &[f64], d: f64) -> Vec<f64> {
        let k = self.k();
        let z = self.z(&self.moments(), x);
        (0..self.cfg.n_targets)
            .map(|j| {
                let (pi, r) = (self.pi(j), self.noise_of(j));
                if self.is_unsized(pi) || r.is_nan() {
                    return f64::INFINITY;
                }
                let p = &self.p[pi];
                let phi: Vec<f64> = (0..k).map(|a| self.phi(a, d)).collect();
                let prior: Vec<f64> = (0..k * k)
                    .map(|rc| p[rc] * phi[rc / k] * phi[rc % k])
                    .collect();
                let gap = self.elapsed[pi] + d;
                let noise: f64 = (0..k)
                    .filter(|&a| p[a * k + a] != 0.0)
                    .map(|a| self.q(r, a) * self.gap2(a, gap) * z[a] * z[a])
                    .sum();
                (1.0 + (quad_of(&prior, &z) + noise) / r).sqrt()
            })
            .collect()
    }
}

/// One row: the features, the targets, the clock step and the weight.
type TwinRow = (Vec<f64>, Vec<Option<f64>>, f64, f64);

/// NaN against NaN, an infinity against itself, and otherwise within
/// `tol` of the larger of `|want|` and `floor`.
fn close(got: f64, want: f64, tol: f64, floor: f64) -> bool {
    if got.is_nan() || want.is_nan() {
        return got.is_nan() && want.is_nan();
    }
    if got.is_infinite() || want.is_infinite() {
        return got == want;
    }
    (got - want).abs() <= tol * want.abs().max(floor)
}

/// The model and its twin fed `rows`: before each row the row
/// statistic, and after it the prediction it made, `pred_var` at
/// `probe`, the coefficients, their variances and the summary's
/// statistic, each held to the twin's; `after` sees the model after
/// each row. Returns how many finite numbers were compared.
fn against_the_twin(
    cfg: KalmanCfg,
    rows: &[TwinRow],
    probe: &[f64],
    what: &str,
    mut after: impl FnMut(usize, &Kalman),
) -> usize {
    let mut m = Kalman::new(cfg.clone()).unwrap();
    let mut twin = Twin::new(cfg);
    let mut compared = 0;
    let mut check = |name: &str, i: usize, got: &[f64], want: &[f64]| {
        assert_eq!(got.len(), want.len(), "{what}: row {i}, {name}");
        for (s, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                close(*g, *w, 1e-9, 1e-3),
                "{what}: row {i}, {name}[{s}]: {g} against {w}"
            );
            compared += usize::from(w.is_finite());
        }
    };
    let mut out = Vec::new();
    for (i, (x, ys, d, w)) in rows.iter().enumerate() {
        m.row_error_inflation_into(x, *d, &mut out);
        check("row statistic", i, &out, &twin.row_stat(x, *d));
        let pred = m.step(x, ys, *d, *w).pred;
        check("pred", i, &pred, &twin.step(x, ys, *d, *w));
        after(i, &m);
        check("pred_var", i, &m.pred_var(probe), &twin.pred_var(probe));
        check(
            "coef",
            i,
            &m.coefficients().concat(),
            &twin.coefficients().concat(),
        );
        let Some(crate::CoefVariance::Absolute(v)) = m.coef_variance() else {
            panic!("kalman's variances are absolute");
        };
        check(
            "coef variance",
            i,
            &v.concat(),
            &twin.coef_variance().concat(),
        );
        m.error_inflation_into(&mut out);
        check("summary", i, &out, &twin.summary());
    }
    compared
}

/// Every readout is its definition, on a stream that moves under the
/// anchor (docs/PLAN.md task 218): one feature's mean drifts a twentieth
/// of a scale a row and the other's spread grows by a hundredth, so the
/// map from the anchor to the moments is seldom the identity and the
/// anchor is re-mapped as they drift past it; the first target is null one row
/// in five and the second present one row in three, so each covariance
/// carries the noise of the clock since its target's last row into
/// `pred_var`, `se_coef` and the summary, and the second target has no
/// residual variance until row 3, which `share_p` reads at the mean;
/// its first value is an innovation of exactly 0, which sizes no noise;
/// a weight of 0 one row in nine, steps of 3. Unshared and shared, with
/// and without an intercept, standardized and not, as a random walk and
/// with slots reverting at half-lives of their own -- the intercept's
/// too, where the transition moves the intercept's variance through
/// the coupling.
#[test]
fn every_readout_is_its_recursion_on_a_moving_stream() {
    let inf = f64::INFINITY;
    let mut s = 218u64;
    let rows: Vec<TwinRow> = (0..160usize)
        .map(|i| {
            let t = i as f64;
            let x = vec![2.0 + 0.05 * t + lcg(&mut s), (0.3 + 0.01 * t) * lcg(&mut s)];
            let y0 = (i % 5 != 3).then(|| 0.5 + x[0] - 2.0 * x[1] + 0.3 * lcg(&mut s));
            let y1 = match i {
                0 => Some(0.0),
                _ if i % 3 == 0 => Some(-x[0] + x[1] + 0.6 * lcg(&mut s)),
                _ => None,
            };
            let d = match i {
                0 => 0.0,
                _ if i % 7 == 2 => 3.0,
                _ => 1.0,
            };
            let w = if i % 9 == 5 {
                0.0
            } else {
                0.5 + (lcg(&mut s) + 1.0)
            };
            (x, vec![y0, y1], d, w)
        })
        .collect();
    for (standardize, fit_intercept, share, revert) in [
        (true, true, false, vec![inf]),
        (true, true, false, vec![30.0, 40.0, 8.0]),
        (true, true, true, vec![inf, 25.0, 60.0]),
        (true, false, false, vec![inf]),
        (true, false, true, vec![inf, 40.0]),
        (false, true, false, vec![inf, 40.0, 8.0]),
        (false, true, true, vec![inf]),
    ] {
        let mut c = cfg(2, 2, vec![20.0]);
        c.standardize = standardize;
        c.fit_intercept = fit_intercept;
        c.share_p = share;
        c.revert_half_life = revert.clone();
        c.decay = Decay::Lam(0.9);
        c.min_weight = 0.0;
        c.p0 = 2.0;
        let what = format!(
            "standardize {standardize}, intercept {fit_intercept}, share {share}, \
             revert {revert:?}"
        );
        // The case each check is for, reached: a covariance read across
        // a gap in its target while the map has moved off the identity.
        let mut gaps = 0;
        let compared = against_the_twin(c, &rows, &[1.7, -0.2], &what, |_, m| {
            let moved = !standardize
                || m.map_c
                    .iter()
                    .chain(&m.map_a)
                    .any(|v| *v != 0.0 && *v != 1.0);
            gaps += usize::from(m.elapsed.iter().any(|e| *e > 0.0) && moved);
        });
        assert!(gaps > 25, "{what}: {gaps} rows read across a gap");
        assert!(compared > 2000, "{what}: {compared} numbers compared");
    }
}

/// A slot read at exactly its mean on the row that sizes it takes no
/// correction there, so its coefficient stays 0 and its covariance with
/// every other slot 0, its variance alone sized (docs/PLAN.md task 218).
/// It holds something all the same: it is read against its anchor, it
/// takes the process noise on the next row that observes the target, its
/// gap noise reaches `pred_var` across a row that does not, and it is not
/// sized a second time -- which only a filter with no process noise shows,
/// the noise on the next row putting a covariance into the intercept's row
/// once the mean has moved. The first two rows put the mean at 2 to the
/// bit, `1 + (0.9 / 1.8) (3 − 1)`, and the third reads 2 there.
#[test]
fn a_slot_read_at_its_mean_on_the_row_that_sizes_it_holds_its_variance() {
    for (half_life, fourth) in [(5.0, None), (f64::INFINITY, Some(2.2))] {
        let mut c = cfg(1, 1, vec![half_life]);
        c.decay = Decay::Lam(0.9);
        c.min_weight = 0.0;
        let mut s = 11u64;
        let mut rows: Vec<TwinRow> = vec![
            (vec![1.0], vec![Some(2.0)], 0.0, 1.0),
            (vec![3.0], vec![Some(3.5)], 1.0, 0.9),
            (vec![2.0], vec![Some(1.0)], 1.0, 1.0),
            (vec![2.5], vec![fourth], 1.0, 1.0),
            (vec![1.5], vec![Some(0.7)], 1.0, 1.0),
        ];
        for i in 0..40 {
            let x = 2.0 + lcg(&mut s);
            let y = (i % 4 != 2).then(|| 1.0 + 0.8 * x + 0.2 * lcg(&mut s));
            rows.push((vec![x], vec![y], 1.0, 1.0));
        }
        let what = format!("a slot read at its mean, half-life {half_life}");
        let mut reached = false;
        let compared = against_the_twin(c, &rows, &[2.6], &what, |i, m| {
            if i == 2 {
                assert_eq!(m.stats.deviation(1, 2.0), 0.0, "the row read the mean");
                assert_eq!((m.beta[0][1], m.p[0][1], m.p[0][2]), (0.0, 0.0, 0.0));
                assert!(m.p[0][3] > 0.0, "sized: {:?}", m.p[0]);
                reached = true;
            }
        });
        assert!(reached, "{what}");
        assert!(compared > 150, "{what}: {compared} numbers compared");
    }
}

/// Under `share_p`, an innovation of exactly 0 sizes no prior
/// (docs/PLAN.md task 218, the module doc): a row's square in the noise
/// basis is the mean over the targets it observes whose squares give a
/// scale. The second target is 0 on the first three rows, where every
/// coefficient is still 0, so its innovations are exactly 0, and the prior
/// rests on the first target's squares alone -- counted, they would halve
/// each row's mean square, and the prior with them.
#[test]
fn under_share_p_an_innovation_of_exactly_0_sizes_no_prior() {
    let mut c = cfg(1, 2, vec![20.0]);
    c.share_p = true;
    c.decay = Decay::Lam(0.9);
    c.min_weight = 0.0;
    let mut s = 41u64;
    let rows: Vec<TwinRow> = (0..60usize)
        .map(|i| {
            let x = vec![lcg(&mut s)];
            let y0 = 1.0 + 2.0 * x[0] + 0.3 * lcg(&mut s);
            let y1 = if i < 3 {
                0.0
            } else {
                -x[0] + 0.3 * lcg(&mut s)
            };
            let d = if i == 0 { 0.0 } else { 1.0 };
            (x, vec![Some(y0), Some(y1)], d, 1.0)
        })
        .collect();
    let mut reached = false;
    let compared = against_the_twin(c, &rows, &[0.4], "a target at 0", |i, m| {
        if i == 2 {
            assert_eq!(m.basis[0].e2.len(), 3, "the prior's three rows");
            assert!(m.p[0][0] > 0.0, "sized: {:?}", m.p[0]);
            reached = true;
        }
    });
    assert!(reached);
    assert!(compared > 500, "{compared} numbers compared");
}

/// A column that holds nothing reads no anchor (docs/PLAN.md task 218):
/// a feature at 0 for the first 30 rows, unsized beside one that is
/// sized, jumps to 5000 with a spread of 100, so that its moments move
/// far past `REMAP_LIMIT` from an anchor it never had. Its anchor is
/// set on the row that sizes it, and the jump moves no slot that holds
/// something: a column counted as holding something there would have
/// refused the re-map whole and moved the anchor of the slot that does
/// under its numbers. The target is null one row in six, so `pred_var`
/// reads the gap noise of a sized slot beside the unsized one.
#[test]
fn a_column_that_holds_nothing_reads_no_anchor() {
    let mut c = cfg(2, 1, vec![20.0]);
    c.decay = Decay::Lam(0.9);
    c.min_weight = 0.0;
    let mut s = 29u64;
    let rows: Vec<TwinRow> = (0..90usize)
        .map(|i| {
            let x = vec![
                lcg(&mut s),
                if i < 30 {
                    0.0
                } else {
                    5000.0 + 100.0 * lcg(&mut s)
                },
            ];
            let y = (i % 6 != 4).then(|| 1.0 + x[0] + 0.002 * (x[1] - 5000.0) + 0.1 * lcg(&mut s));
            (x, vec![y], if i == 0 { 0.0 } else { 1.0 }, 1.0)
        })
        .collect();
    let mut seen = [false; 2];
    let compared = against_the_twin(c, &rows, &[0.3, 5050.0], "a column at 0", |i, m| {
        if i == 29 {
            assert!(m.p[0][4] > 0.0 && m.p[0][8] == 0.0, "{:?}", m.p[0]);
            seen[0] = true;
        }
        if i == 31 {
            assert!(m.p[0][8] > 0.0, "sized once it moved: {:?}", m.p[0]);
            seen[1] = true;
        }
    });
    assert_eq!(seen, [true; 2]);
    assert!(compared > 500, "{compared} numbers compared");
}

/// A damaged state is refused as it is read, each part on its own
/// (docs/PLAN.md task 218): a list one covariance or one target short,
/// an anchor of the wrong length, an anchor scale that is 0, below 0 or
/// infinite, an anchor mean that is not finite, a clock since an
/// observation that is below 0 or infinite, and a noise basis whose two
/// lists differ in length, that holds more than `PRIOR_ROWS` rows, or
/// whose squares or weights hold a 0, a number below 0 or an infinity.
/// Each would be read later as a number the filter never makes. The
/// control loads.
#[test]
fn a_state_damaged_in_any_one_part_is_refused() {
    use rmpv::Value;
    fn field<'a>(v: &'a mut Value, name: &str) -> &'a mut Value {
        let Value::Map(entries) = v else {
            panic!("a map")
        };
        &mut entries
            .iter_mut()
            .find(|(k, _)| k.as_str() == Some(name))
            .unwrap_or_else(|| panic!("no {name}"))
            .1
    }
    fn list(v: &mut Value) -> &mut Vec<Value> {
        let Value::Array(items) = v else {
            panic!("a list")
        };
        items
    }
    // Two targets, so two covariances and two bases, each holding two
    // rows' squares.
    let mut c = cfg(2, 2, vec![50.0]);
    c.min_weight = 0.0;
    let m = fit(c, 2, 7);
    assert!(m.basis.iter().all(|b| b.e2.len() == 2), "{:?}", m.basis);
    assert!(reread(&m, |_| {}).is_ok(), "the control");
    let set = |name: &'static str, at: usize, to: f64| {
        move |v: &mut Value| list(field(v, name))[at] = Value::F64(to)
    };
    let basis = |edit: fn(&mut Value)| move |v: &mut Value| edit(&mut list(field(v, "basis"))[0]);
    type Edit = Box<dyn Fn(&mut Value)>;
    let (shape, anchor, clocks, noise) = (
        "kalman: state has the wrong shape",
        "the anchor's scales must be finite and > 0, its means finite",
        "kalman: the clocks since an observation must be finite and >= 0",
        "kalman: the noise basis must hold at most 3 rows, as many weights as squared \
         innovations, each finite and > 0",
    );
    let mut cases: Vec<(String, Edit, &str)> = Vec::new();
    for name in ["beta", "p", "elapsed", "basis", "sig2", "wsig", "wj"] {
        cases.push((
            format!("{name} short"),
            Box::new(move |v| shorten(v, name)),
            shape,
        ));
    }
    for name in ["anchor_hi", "anchor_lo", "anchor_scale"] {
        cases.push((
            format!("{name} short"),
            Box::new(move |v| shorten(v, name)),
            "the state's anchor does not match its cfg's standardize",
        ));
    }
    for (what, edit, says) in [
        ("an anchor scale of 0", set("anchor_scale", 1, 0.0), anchor),
        (
            "an anchor scale below 0",
            set("anchor_scale", 2, -1.0),
            anchor,
        ),
        (
            "an infinite anchor scale",
            set("anchor_scale", 1, f64::INFINITY),
            anchor,
        ),
        (
            "an infinite anchor mean",
            set("anchor_hi", 1, f64::INFINITY),
            anchor,
        ),
        (
            "a low part not a number",
            set("anchor_lo", 2, f64::NAN),
            anchor,
        ),
        ("a clock below 0", set("elapsed", 0, -1.0), clocks),
        (
            "an infinite clock",
            set("elapsed", 1, f64::INFINITY),
            clocks,
        ),
    ] {
        cases.push((what.into(), Box::new(edit), says));
    }
    type Damage = (&'static str, fn(&mut Value));
    let damaged: [Damage; 7] = [
        ("one square more than weights", |b| {
            list(field(b, "e2")).push(Value::F64(1.0))
        }),
        ("four rows", |b| {
            for name in ["e2", "w"] {
                let items = list(field(b, name));
                items.push(Value::F64(1.0));
                items.push(Value::F64(1.0));
            }
        }),
        ("a square of 0", |b| {
            list(field(b, "e2"))[0] = Value::F64(0.0)
        }),
        ("a weight of 0", |b| {
            list(field(b, "w"))[1] = Value::F64(0.0)
        }),
        ("a square below 0", |b| {
            list(field(b, "e2"))[1] = Value::F64(-2.0)
        }),
        ("an infinite square", |b| {
            list(field(b, "e2"))[0] = Value::F64(f64::INFINITY)
        }),
        ("an infinite weight", |b| {
            list(field(b, "w"))[0] = Value::F64(f64::INFINITY)
        }),
    ];
    for (what, edit) in damaged {
        cases.push((format!("a basis with {what}"), Box::new(basis(edit)), noise));
    }
    for (what, edit, says) in &cases {
        let err = reread(&m, |v| edit(v)).expect_err(what);
        assert!(err.contains(says), "{what}: {err}");
    }
    assert_eq!(cases.len(), 24);
}

/// The re-map is refused whole where it would move the coordinates past
/// `REMAP_LIMIT` or leave a number that is not finite -- each `b'`,
/// `P'`'s diagonal and `P'_00` (the module doc) -- and made otherwise
/// (docs/PLAN.md task 218). Refused, `b` and `P` keep their bits; either
/// way the anchor moves to the moments. Each case is a state a file can
/// hold: a filter fitted on a short stream, its `b`, `P` and anchor set
/// so that the map from the anchor to the moments is `a` and `c`, and
/// each refusal turns on one number alone -- `b_1 a`, `b_0 + c b_1`,
/// `P_11 a²`, a scale past the limit -- beside the cases on either side.
#[test]
fn the_re_map_is_refused_whole_past_its_limit_or_a_number() {
    let mut base = Kalman::new({
        let mut c = cfg(1, 1, vec![50.0]);
        c.decay = Decay::Lam(0.9);
        c.min_weight = 0.0;
        c
    })
    .unwrap();
    let mut s = 3u64;
    for i in 0..12 {
        let x = [lcg(&mut s)];
        base.step(
            &x,
            &[Some(1.0 + x[0] + 0.1 * lcg(&mut s))],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    let ident = [1.0, 0.0, 0.0, 1.0];
    let big = 1e308;
    type Case = (&'static str, [f64; 2], [f64; 4], f64, f64, bool);
    let cases: [Case; 10] = [
        (
            "a re-map within bounds",
            [1.0, 2.0],
            [2.0, 0.5, 0.5, 1.0],
            3.0,
            0.5,
            false,
        ),
        ("b_1 a overflows", [1.0, 1e306], ident, 600.0, 0.5, true),
        (
            "b_0 + c b_1 overflows",
            [1.5 * big, 0.8 * big],
            ident,
            1.0,
            1.5,
            true,
        ),
        (
            "b_0 + c b_1 just overflows",
            [0.5 * big, 0.7 * big],
            ident,
            1.0,
            2.0,
            true,
        ),
        (
            "b_0 c b_1 would overflow",
            [1e200, 1e200],
            ident,
            1.0,
            2.0,
            false,
        ),
        (
            "P_11 a² overflows",
            [1.0, 1.0],
            [1e290, 1e289, 1e289, 1e303],
            600.0,
            0.35,
            true,
        ),
        (
            "a scale past the limit",
            [1.0, 1.0],
            ident,
            2000.0,
            0.5,
            true,
        ),
        (
            "a scale shrunk past it",
            [1.0, 1.0],
            ident,
            1.0 / 2000.0,
            0.5,
            true,
        ),
        ("a mean past it", [1.0, 1.0], ident, 1.0, 2000.0, true),
        ("a mean at it", [1.0, 1.0], ident, 1.0, 1000.0, false),
    ];
    for (what, b, p, a, c, refused) in cases {
        let mut m = base.clone();
        let (hi, lo) = m.stats.mean_pair(1);
        let scale = m.scales()[1];
        m.anchor_scale[1] = scale / a;
        m.anchor_hi[1] = hi - c * m.anchor_scale[1];
        m.anchor_lo[1] = lo;
        m.beta[0] = b.to_vec();
        m.p[0] = p.to_vec();
        // The map as the module doc defines it, from that anchor.
        let sa = m.anchor_scale[1];
        let (a, c) = (
            scale / sa,
            ((hi - m.anchor_hi[1]) + (lo - m.anchor_lo[1])) / sa,
        );
        let b_new = [b[0] + c * b[1], a * b[1]];
        let u = [p[0] + c * p[2], p[1] + c * p[3]];
        let p_new = [u[0] + c * u[1], a * u[1], a * u[1], a * a * p[3]];
        let within = (1.0 / REMAP_LIMIT..=REMAP_LIMIT).contains(&a) && c.abs() <= REMAP_LIMIT;
        let finite = b_new
            .iter()
            .chain([&p_new[0], &p_new[3]])
            .all(|v| v.is_finite());
        assert_eq!(
            !(within && finite),
            refused,
            "{what}: the case is what it says"
        );
        m.mark_live();
        m.follow_the_moments();
        if refused {
            assert_eq!(m.beta[0], b, "{what}: b kept");
            assert_eq!(m.p[0], p, "{what}: P kept");
        } else {
            for (got, want) in m.beta[0]
                .iter()
                .zip(&b_new)
                .chain(m.p[0].iter().zip(&p_new))
            {
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs(),
                    "{what}: {got} against {want}"
                );
            }
        }
        assert_eq!(
            (m.anchor_hi[1], m.anchor_lo[1], m.anchor_scale[1]),
            (hi, lo, scale),
            "{what}: the anchor moved"
        );
    }
}

/// The anchor moves to the moments on the row a slot that holds
/// anything drifts past `ANCHOR_DRIFT` from it -- a scale by more than a
/// factor of 2 either way, `max(a, 1/a) − 1 > 1`, or a mean by more than
/// one of the anchor's scales, `|c| > 1` -- and stays put on every other
/// row, the map to it held as the module doc defines it: `a = s / s^a`,
/// `c = ((m_hi − m^a_hi) + (m_lo − m^a_lo)) / s^a`, the means as the
/// pairs they are kept as (docs/PLAN.md task 218). One feature sits at a
/// level of 2^20 with a spread near 2^-12, where a mean's low part is a
/// sizeable part of a scale -- up to 2^-33 of 2^-12 -- and its mean
/// drifts; the other's spread grows twenty-fold and then shrinks.
#[test]
fn the_anchor_moves_on_the_row_the_drift_passes_its_bound() {
    let mut c = cfg(2, 1, vec![30.0]);
    c.decay = Decay::Lam(0.8);
    c.min_weight = 0.0;
    let mut m = Kalman::new(c).unwrap();
    let (k, level, unit) = (3, 1_048_576.0, 2f64.powi(-12));
    let mut s = 218u64;
    let (mut moved, mut held, mut low_parts) = (0, 0, 0);
    for i in 0..400usize {
        let t = i as f64;
        let spread = if i < 200 {
            1.0 + 0.1 * t
        } else {
            21.0 * 0.98f64.powi(i as i32 - 200)
        };
        let x = [
            level + unit * (lcg(&mut s) + 0.05 * t),
            spread * lcg(&mut s),
        ];
        let y = (i % 4 != 1)
            .then(|| 1.0 + (x[0] - level) / unit - 0.5 * x[1] / spread + 0.1 * lcg(&mut s));
        let w = if i % 11 == 7 { 0.0 } else { 1.0 };
        let anchor = (
            m.anchor_hi.clone(),
            m.anchor_lo.clone(),
            m.anchor_scale.clone(),
        );
        let sized: Vec<bool> = (0..k).map(|a| m.p[0][a * k + a] != 0.0).collect();
        m.step(&x, &[y], if i == 0 { 0.0 } else { 1.0 }, w);
        if (0..k).any(|a| !sized[a] && m.p[0][a * k + a] != 0.0) {
            continue; // a slot sized on the row takes its anchor there
        }
        let mut drift = 0.0f64;
        let mut map = Vec::new();
        for a in 1..k {
            if m.p[0][a * k + a] == 0.0 && m.beta[0][a] == 0.0 {
                continue;
            }
            let v = m.stats.var(a);
            let now = if v > 0.0 && v.is_finite() {
                v.sqrt()
            } else {
                1.0
            };
            let (hi, lo) = m.stats.mean_pair(a);
            let ra = now / anchor.2[a];
            let rc = ((hi - anchor.0[a]) + (lo - anchor.1[a])) / anchor.2[a];
            drift = drift.max(ra - 1.0).max(1.0 / ra - 1.0).max(rc.abs());
            low_parts += usize::from(a == 1 && lo != anchor.1[a]);
            map.push((a, ra, rc, now, hi, lo));
        }
        if w > 0.0 && drift > ANCHOR_DRIFT {
            for &(a, _, _, now, hi, lo) in &map {
                assert_eq!(
                    (m.anchor_hi[a], m.anchor_lo[a], m.anchor_scale[a]),
                    (hi, lo, now),
                    "row {i}: drift {drift}, slot {a} re-anchored"
                );
            }
            moved += 1;
        } else {
            assert_eq!(
                (&m.anchor_hi, &m.anchor_lo, &m.anchor_scale),
                (&anchor.0, &anchor.1, &anchor.2),
                "row {i}: drift {drift}, the anchor held"
            );
            for &(a, ra, rc, ..) in &map {
                assert_eq!(
                    (m.map_a[a].to_bits(), m.map_c[a].to_bits()),
                    (ra.to_bits(), rc.to_bits()),
                    "row {i}, slot {a}: the map {} {} against {ra} {rc}",
                    m.map_a[a],
                    m.map_c[a]
                );
            }
            held += usize::from(!map.is_empty());
        }
    }
    assert!(
        moved > 10 && held > 200,
        "{moved} rows re-anchored, {held} held"
    );
    assert!(low_parts > 100, "{low_parts} rows read a low part");
}

/// The intercept in the caller's units reads the anchor's mean whole,
/// `c_0 = b_0 − Σ_i c_i (m^a_hi + m^a_lo)` (the module doc; docs/PLAN.md
/// task 218): on an anchor at `2^30 + 2^-24`, the low part a quarter of
/// a rounding step of the high, and a slope of 8, the intercept of a
/// fit whose `b_0` is `8 · 2^30 + 3` is `3 − 8 · 2^-24`, every step
/// exact. The low part is all that the intercept has left of the mean.
#[test]
fn the_intercept_reads_the_anchors_mean_whole() {
    let mut m = Kalman::new(cfg(1, 1, vec![50.0])).unwrap();
    let (hi, lo) = (2f64.powi(30), 2f64.powi(-24));
    assert!(lo < 0.5 * (hi.next_up() - hi), "a pair as `comp` keeps it");
    m.anchor_hi[1] = hi;
    m.anchor_lo[1] = lo;
    m.anchor_scale[1] = 0.5;
    m.beta[0] = vec![8.0 * hi + 3.0, 4.0];
    assert_eq!(m.coefficients()[0], vec![3.0 - 8.0 * lo, 8.0]);
}
