//! The filter against its recursion written out by hand, every branch of a
//! row: the old form (`P` absolute, the noise estimate in the gain) and,
//! since task 278, `P` in units of the noise with a diffuse intercept. A
//! file of its own because `kalman.rs` stands near the source-size cap
//! (`tests/test_repo_hygiene.py`).

use super::*;

/// The standardized filter is its recursion, written from the module
/// doc with every moment from its definition (`the_filter_is_its_recursion`
/// holds the unstandardized one): the row standardized against the EW
/// means and variances of the rows before it, `z_i = (x_i − m_i) / s_i`
/// (`x_i / s_i`, the root mean square, through the origin), each a
/// weighted sum over the history with the weights aged by the decay; the
/// filter's recursion on `z` in those, the current, coordinates, with the
/// transition on every row and the process noise `Q D²` on each row that
/// observes the covariance's target, `D` the clock since the last; each
/// slot's prior `p0 R` on the first observed row its scale is usable, `R`
/// the mean squared innovation once three rows give it; and after every
/// row that moves the moments, the change of coordinates from the moments
/// before it to the moments after it, `A` with `A_00 = 1`, `A_0i = (m'_i −
/// m_i) / s_i`, `A_ii = s'_i / s_i`, applied as `b' = A b` and `P' = A P
/// Aᵀ` by matrix products -- task 206's re-map on every row, which the
/// anchored filter is to rounding (task 211). Two targets, one present
/// one row in three, weights other than 1 and 0, steps of 1 and 3, shared
/// and not -- shared, the noise is the mean residual variance over the
/// targets that have one (review round 5, A1) -- with and without an
/// intercept, and with slopes reverting at half-lives of their own. The
/// predictions agree to 1e-12 of their size.
///
/// And in units of the noise (`noise_units`, task 278): `P` sized at `p0`
/// on each slope once its scale is usable, needing no noise; the gain
/// `P z / (zᵀ P z + 1 / w)`, reading none; the process noise `(ln 2 /
/// h)² D²`; and on the first row that observes a covariance's target, the
/// intercept diffuse: `b_0` takes the whole innovation and `P` the exact
/// step, `P_00 = zᵀ P z + 1 / w`, `P_0i = −(P z)_i`, the slopes unmoved. A
/// target first seen later has its intercept centred the same way.
#[test]
fn the_standardized_filter_is_its_recursion() {
    let inf = f64::INFINITY;
    for (units, fit_intercept, share, revert) in [false, true].into_iter().flat_map(|u| {
        [
            (u, true, false, vec![inf]),
            (u, true, true, vec![inf]),
            (u, false, false, vec![inf]),
            (u, true, false, vec![inf, 40.0, 8.0]),
            (u, true, true, vec![inf, 25.0, 60.0]),
        ]
    }) {
        let hq = 40.0;
        let decay = Decay::Lam(0.9);
        let mut c = cfg(2, 2, vec![hq]);
        c.fit_intercept = fit_intercept;
        c.share_p = share;
        c.min_weight = 0.0;
        c.decay = decay;
        c.revert_half_life = revert.clone();
        c.noise_units = units;
        let mut m = Kalman::new(c.clone()).unwrap();
        let off = usize::from(fit_intercept);
        let k = 2 + off;
        let n_p = if share { 1 } else { 2 };
        let mut p = vec![vec![0.0f64; k * k]; n_p];
        let mut b = vec![vec![0.0f64; k]; 2];
        let (mut sig2, mut wsig, mut wj) = ([0.0f64; 2], [0.0f64; 2], [0.0f64; 2]);
        let mut elapsed = vec![0.0f64; n_p];
        // Per covariance: Σ w e², Σ w and the rows.
        // Per covariance, the first three rows' mean squared
        // innovation and weight (task 214).
        let mut basis: Vec<Vec<(f64, f64)>> = vec![Vec::new(); n_p];
        let mut history: Vec<([f64; 2], f64)> = Vec::new();
        // The moments of the history, by their definition: `(m, s,
        // usable)` per feature, `m = 0` through the origin.
        let moments = |h: &[([f64; 2], f64)]| -> ([f64; 2], [f64; 2], [bool; 2]) {
            let total: f64 = h.iter().map(|(_, w)| w).sum();
            let (mut mean, mut scale, mut usable) = ([0.0; 2], [1.0; 2], [false; 2]);
            if total <= 0.0 {
                return (mean, scale, usable);
            }
            for f in 0..2 {
                if fit_intercept {
                    let mu = h.iter().map(|(x, w)| w * x[f]).sum::<f64>() / total;
                    let var = h.iter().map(|(x, w)| w * (x[f] - mu).powi(2)).sum::<f64>() / total;
                    mean[f] = mu;
                    usable[f] = var > 0.0;
                    scale[f] = if usable[f] { var.sqrt() } else { 1.0 };
                } else {
                    let raw = h.iter().map(|(x, w)| w * x[f] * x[f]).sum::<f64>() / total;
                    usable[f] = raw > 0.0;
                    scale[f] = if usable[f] { raw.sqrt() } else { 1.0 };
                }
            }
            (mean, scale, usable)
        };
        let phi_of = |i: usize, d: f64| -> f64 {
            let r = if revert.len() == 1 {
                revert[0]
            } else {
                revert[i]
            };
            Decay::Halflife(r).factor(d)
        };
        let mut s = 211u64;
        let mut checked = 0;
        for i in 0..150 {
            let x = [2.0 + lcg(&mut s), 0.3 * lcg(&mut s)];
            let ys = [
                Some(0.5 + x[0] - 2.0 * x[1] + 0.2 * lcg(&mut s)),
                (i % 3 == 0).then(|| -x[0] + 3.0 * lcg(&mut s)),
            ];
            let w = if i % 9 == 5 {
                0.0
            } else {
                0.5 + (lcg(&mut s) + 1.0)
            };
            let d = match i {
                0 => 0.0,
                _ if i % 7 == 2 => 3.0,
                _ => 1.0,
            };
            let lam = decay.factor(d);
            for e in elapsed.iter_mut() {
                *e += d;
            }
            // The transition, in the current coordinates.
            let phi: Vec<f64> = (0..k).map(|a| phi_of(a, d)).collect();
            for bj in b.iter_mut() {
                for a in 0..k {
                    bj[a] *= phi[a];
                }
            }
            for pp in p.iter_mut() {
                for a in 0..k {
                    for bb in 0..k {
                        pp[a * k + bb] *= phi[a] * phi[bb];
                    }
                }
            }
            let (mean, scale, usable) = moments(&history);
            let mut z = vec![1.0; k];
            for f in 0..2 {
                z[off + f] = (x[f] - mean[f]) / scale[f];
            }
            let dotz = |v: &[f64]| -> f64 { (0..k).map(|a| z[a] * v[a]).sum() };
            // A covariance not yet sized predicts nothing (task 270,
            // review round 9 B-1).
            let sized = |j: usize| {
                let pj = &p[if share { 0 } else { j }];
                (0..k).any(|a| pj[a * k + a] != 0.0)
            };
            let want: Vec<f64> = (0..2)
                .map(|j| {
                    if wj[j] > 0.0 && sized(j) {
                        dotz(&b[j])
                    } else {
                        f64::NAN
                    }
                })
                .collect();
            let got = m.step(&x, &ys, d, w).pred;
            for j in 0..2 {
                assert_eq!(
                    got[j].is_nan(),
                    want[j].is_nan(),
                    "{fit_intercept} {share} {revert:?}, row {i}"
                );
                if want[j].is_finite() {
                    assert!(
                        (got[j] - want[j]).abs() <= 1e-12 * want[j].abs().max(1.0),
                        "intercept {fit_intercept}, share {share}, revert {revert:?}, row \
                         {i}, target {j}: {} against {}",
                        got[j],
                        want[j]
                    );
                    checked += 1;
                }
            }
            // The filter on `z`, as `the_filter_is_its_recursion` writes
            // it: a row of weight 0 observes nothing.
            let obs: Vec<Option<f64>> = ys.iter().map(|y| y.filter(|_| w > 0.0)).collect();
            let e2: Vec<Option<f64>> = (0..2)
                .map(|j| obs[j].map(|y| (y - dotz(&b[j])).powi(2)))
                .collect();
            let seen: Vec<f64> = e2.iter().flatten().copied().collect();
            let shared_first = if seen.is_empty() {
                0.0
            } else {
                seen.iter().sum::<f64>() / seen.len() as f64
            };
            let have: Vec<f64> = sig2.iter().copied().filter(|v| *v > 0.0).collect();
            let shared = if have.is_empty() {
                0.0
            } else {
                have.iter().sum::<f64>() / have.len() as f64
            };
            let noise: Vec<f64> = (0..2)
                .map(|j| {
                    let s2 = if share { shared } else { sig2[j] };
                    if s2 > 0.0 {
                        s2
                    } else if share {
                        shared_first
                    } else {
                        e2[j].unwrap_or(0.0)
                    }
                })
                .collect();
            // A row counts once, whatever the targets it observes.
            for (pi, held) in basis.iter_mut().enumerate() {
                let mine: Vec<f64> = (0..2)
                    .filter(|&j| share || j == pi)
                    .filter_map(|j| e2[j].filter(|v| *v > 0.0))
                    .collect();
                if !mine.is_empty() && held.len() < 3 {
                    held.push((mine.iter().sum::<f64>() / mine.len() as f64, w));
                }
            }
            for pi in 0..n_p {
                let informs = if share {
                    obs.iter().any(Option::is_some)
                } else {
                    obs[pi].is_some()
                };
                if !informs {
                    continue;
                }
                let gap = elapsed[pi];
                elapsed[pi] = 0.0;
                // In units of the noise the process noise reads none.
                let q = if units { 1.0 } else { noise[pi] } * (std::f64::consts::LN_2 / hq).powi(2);
                for a in 0..k {
                    // `D²`, or on a reverting slot `((1 − 2^(−D/r)) /
                    // θ)²`, `θ = ln 2 / r` (task 214).
                    let r = if revert.len() == 1 {
                        revert[0]
                    } else {
                        revert[a]
                    };
                    let dd = if r.is_infinite() {
                        gap * gap
                    } else {
                        let theta = std::f64::consts::LN_2 / r;
                        ((1.0 - phi_of(a, gap)) / theta).powi(2)
                    };
                    if p[pi][a * k + a] != 0.0 {
                        p[pi][a * k + a] += q * dd;
                    }
                }
                if units {
                    // `p0`, 1 here, on each slope once its scale is usable;
                    // the intercept waits for its diffuse step.
                    for a in off..k {
                        if usable[a - off] && p[pi][a * k + a] == 0.0 {
                            p[pi][a * k + a] = 1.0;
                        }
                    }
                } else if basis[pi].len() == 3 {
                    // The weighted median: the value with less than
                    // half the weight below it and at least half at or
                    // below it, over the median of a chi-squared of one
                    // degree, 0.4549364231195728.
                    let held = &basis[pi];
                    let total: f64 = held.iter().map(|(_, w)| w).sum();
                    let below = |v: f64| -> f64 {
                        held.iter().filter(|(u, _)| *u < v).map(|(_, w)| w).sum()
                    };
                    let at_or_below = |v: f64| -> f64 {
                        held.iter().filter(|(u, _)| *u <= v).map(|(_, w)| w).sum()
                    };
                    let median = held
                        .iter()
                        .map(|(v, _)| *v)
                        .filter(|&v| below(v) < 0.5 * total && at_or_below(v) >= 0.5 * total)
                        .fold(f64::INFINITY, f64::min);
                    let r = median / 0.4549364231195728;
                    for a in 0..k {
                        let ok = a < off || usable[a - off];
                        if ok && p[pi][a * k + a] == 0.0 {
                            p[pi][a * k + a] = r;
                        }
                    }
                }
            }
            let mut shared_row: Option<(Vec<f64>, f64)> = None;
            // The exact step of a prior diffuse in the intercept: `P_00 = s`,
            // `P_0i = −(P z)_i`, the slopes' block unmoved.
            let diffuse = |pp: &mut Vec<f64>, pz: &[f64], s_inn: f64| {
                pp[0] = s_inn;
                for a in 1..k {
                    pp[a] = -pz[a];
                    pp[a * k] = -pz[a];
                }
            };
            let open: Vec<bool> = (0..n_p)
                .map(|pi| units && fit_intercept && p[pi][0] == 0.0)
                .collect();
            for j in 0..2 {
                let pi = if share { 0 } else { j };
                let sigma2 = noise[j];
                let Some(y) = obs[j] else {
                    wj[j] *= lam;
                    wsig[j] *= lam;
                    continue;
                };
                let pz: Vec<f64> = (0..k).map(|a| dotz(&p[pi][a * k..(a + 1) * k])).collect();
                let r_inn = if units { 1.0 } else { sigma2 };
                let s_inn = dotz(&pz) + r_inn / w;
                let mut err = y - dotz(&b[j]);
                if r_inn > 0.0 {
                    if units && fit_intercept && (open[pi] || wj[j] == 0.0) {
                        b[j][0] += err;
                        err = 0.0;
                    }
                    if !open[pi] {
                        for a in 0..k {
                            b[j][a] += pz[a] / s_inn * err;
                        }
                    }
                    if share {
                        shared_row = Some((pz, s_inn));
                    } else if open[pi] {
                        diffuse(&mut p[pi], &pz, s_inn);
                    } else {
                        for a in 0..k {
                            for bb in 0..k {
                                p[pi][a * k + bb] -= pz[a] * pz[bb] / s_inn;
                            }
                        }
                    }
                }
                let aged = lam * wsig[j];
                wsig[j] = aged;
                if want[j].is_finite() {
                    let r = y - want[j];
                    sig2[j] = (aged * sig2[j] + w * r * r) / (aged + w);
                    wsig[j] = aged + w;
                }
                wj[j] = lam * wj[j] + w;
            }
            if let Some((pz, s_inn)) = shared_row {
                if open[0] {
                    diffuse(&mut p[0], &pz, s_inn);
                } else {
                    for a in 0..k {
                        for bb in 0..k {
                            p[0][a * k + bb] -= pz[a] * pz[bb] / s_inn;
                        }
                    }
                }
            }
            // The moments learn the row, at its weight, and `b` and `P`
            // follow them.
            for (_, wh) in history.iter_mut() {
                *wh *= lam;
            }
            history.push((x, w));
            if w > 0.0 {
                let (mean2, scale2, _) = moments(&history);
                let mut am = vec![0.0f64; k * k];
                am[0] = 1.0;
                for f in 0..2 {
                    let a = off + f;
                    am[a * k + a] = scale2[f] / scale[f];
                    if fit_intercept {
                        am[a] = (mean2[f] - mean[f]) / scale[f];
                    }
                }
                for bj in b.iter_mut() {
                    let old = bj.clone();
                    for r in 0..k {
                        bj[r] = (0..k).map(|cc| am[r * k + cc] * old[cc]).sum();
                    }
                }
                for pp in p.iter_mut() {
                    let mut ap = vec![0.0f64; k * k];
                    for r in 0..k {
                        for cc in 0..k {
                            ap[r * k + cc] = (0..k).map(|t| am[r * k + t] * pp[t * k + cc]).sum();
                        }
                    }
                    for r in 0..k {
                        for cc in 0..k {
                            pp[r * k + cc] = (0..k).map(|t| ap[r * k + t] * am[cc * k + t]).sum();
                        }
                    }
                }
            }
        }
        assert!(checked > 250, "{checked} predictions checked");
    }
}

/// The filter written from the module docs, unstandardized and without
/// reversion: per target, `R` and the `Q` from `half_life` are both the
/// EW residual variance, under `share_p` the mean over the targets that
/// have one (review round 5, A1), read once a row before any target's
/// update (review round 4, CC3); before
/// there is one, the row's own innovation squared, under `share_p` the
/// mean of the squares over the targets the row observes, and no noise
/// (no `Q`, no correction) where there is none (CC4); `P` is unsized, 0,
/// until a row has a noise, which sets it to `p0` times that noise, and
/// after that takes `Q D²` once on each row that observes its target
/// (any target, under `share_p`), `D` the clock since the last such row,
/// before the first target's update (task 211);
/// a target present at a positive weight corrects `b` and `P` by the gain
/// `P z / (zᵀ P z + R / w)`, under `share_p` every target from `P` as the
/// row found it and `P` once, after them (task 204); the residual
/// variance is the EW mean of the
/// squared out-of-sample errors, its weight ageing on every row. Two
/// targets, one present one row in three, weights other than 1, shared
/// and not (task 158).
#[test]
fn the_filter_is_its_recursion() {
    for share in [true, false] {
        let (h, hq) = (30.0, 50.0);
        let mut c = cfg(2, 2, vec![hq]);
        c.standardize = false;
        c.share_p = share;
        c.min_weight = 0.0;
        c.decay = Decay::Halflife(h);
        let mut m = Kalman::new(c).unwrap();
        let k = 3;
        let n_p = if share { 1 } else { 2 };
        let p0 = 1.0;
        let mut p = vec![vec![0.0f64; k * k]; n_p];
        let mut b = [[0.0f64; 3]; 2];
        let (mut sig2, mut wsig, mut wj) = ([0.0f64; 2], [0.0f64; 2], [0.0f64; 2]);
        let mut elapsed = vec![0.0f64; n_p];
        let mut s = 103u64;
        for i in 0..150 {
            let x = [lcg(&mut s), 1.0 + lcg(&mut s)];
            let ys = [
                Some(0.5 + x[0] - 2.0 * x[1] + 0.2 * lcg(&mut s)),
                (i % 3 == 0).then(|| -x[0] + 3.0 * lcg(&mut s)),
            ];
            let w = 0.5 + (lcg(&mut s) + 1.0);
            let d = if i == 0 { 0.0 } else { 1.0 };
            for e in elapsed.iter_mut() {
                *e += d;
            }
            let lam = 0.5f64.powf(d / h);
            let z = [1.0, x[0], x[1]];
            let dotz = |v: &[f64]| -> f64 { (0..k).map(|a| z[a] * v[a]).sum() };
            // A covariance not yet sized predicts nothing (task 270,
            // review round 9 B-1).
            let sized = |j: usize| {
                let pj = &p[if share { 0 } else { j }];
                (0..k).any(|a| pj[a * k + a] != 0.0)
            };
            let want: Vec<f64> = (0..2)
                .map(|j| {
                    if wj[j] > 0.0 && sized(j) {
                        dotz(&b[j])
                    } else {
                        f64::NAN
                    }
                })
                .collect();
            let got = m.step(&x, &ys, d, w).pred;
            for j in 0..2 {
                assert_eq!(got[j].is_nan(), want[j].is_nan(), "share {share}, row {i}");
                if want[j].is_finite() {
                    assert!(
                        (got[j] - want[j]).abs() <= 1e-10 * want[j].abs().max(1.0),
                        "share {share}, row {i}, target {j}: {} against {}",
                        got[j],
                        want[j]
                    );
                }
            }
            // Each observed target's innovation squared, from the
            // coefficients before the row's first update.
            let e2: Vec<Option<f64>> = (0..2)
                .map(|j| ys[j].map(|y| (y - dotz(&b[j])).powi(2)))
                .collect();
            let seen: Vec<f64> = e2.iter().flatten().copied().collect();
            let shared_first = if seen.is_empty() {
                0.0
            } else {
                seen.iter().sum::<f64>() / seen.len() as f64
            };
            // Under `share_p` every target's noise is the mean residual
            // variance over the targets that have one, as the row
            // arrives, read once, before any target's update moves one
            // (review round 4, CC3; review round 5, A1): target 1 has
            // none until its second present row, row 3.
            let have: Vec<f64> = sig2.iter().copied().filter(|v| *v > 0.0).collect();
            let shared = if have.is_empty() {
                0.0
            } else {
                have.iter().sum::<f64>() / have.len() as f64
            };
            let mut shared_row: Option<(Vec<f64>, f64)> = None;
            for j in 0..2 {
                let pi = if share { 0 } else { j };
                let s2 = if share { shared } else { sig2[j] };
                let sigma2 = if s2 > 0.0 {
                    s2
                } else if share {
                    shared_first
                } else {
                    e2[j].unwrap_or(0.0)
                };
                let informs = if share {
                    ys.iter().any(Option::is_some)
                } else {
                    ys[j].is_some()
                };
                if (!share || j == 0) && informs {
                    let dd = elapsed[pi] * elapsed[pi];
                    elapsed[pi] = 0.0;
                    if p[pi].iter().all(|v| *v == 0.0) {
                        for a in 0..k {
                            p[pi][a * k + a] = p0 * sigma2;
                        }
                    } else {
                        let q = sigma2 * (std::f64::consts::LN_2 / hq).powi(2);
                        for a in 0..k {
                            p[pi][a * k + a] += q * dd;
                        }
                    }
                }
                let Some(y) = ys[j] else {
                    wj[j] *= lam;
                    wsig[j] *= lam;
                    continue;
                };
                // Under `share_p` every target reads `P` as the row
                // found it: the shared update waits for the loop's end.
                let pz: Vec<f64> = (0..k).map(|a| dotz(&p[pi][a * k..(a + 1) * k])).collect();
                let s_inn = dotz(&pz) + sigma2 / w;
                let err = y - dotz(&b[j]);
                if sigma2 > 0.0 {
                    for a in 0..k {
                        b[j][a] += pz[a] / s_inn * err;
                    }
                    if share {
                        shared_row = Some((pz, s_inn));
                    } else {
                        for a in 0..k {
                            for bb in 0..k {
                                p[pi][a * k + bb] -= pz[a] * pz[bb] / s_inn;
                            }
                        }
                    }
                }
                let aged = lam * wsig[j];
                wsig[j] = aged;
                if want[j].is_finite() {
                    let r = y - want[j];
                    sig2[j] = (aged * sig2[j] + w * r * r) / (aged + w);
                    wsig[j] = aged + w;
                }
                wj[j] = lam * wj[j] + w;
            }
            // `P` takes the row once, if any target updated: every one
            // read the same `P z` and the same noise (task 204).
            if let Some((pz, s_inn)) = shared_row {
                for a in 0..k {
                    for bb in 0..k {
                        p[0][a * k + bb] -= pz[a] * pz[bb] / s_inn;
                    }
                }
            }
        }
    }
}
